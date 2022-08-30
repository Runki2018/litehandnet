import time

import munkres
import numpy as np
import torch
from collections import defaultdict
from utils.bbox_metric import xywh2xyxy, box_iou, bbox_iou
import torchvision

from config import DATASET, pcfg, config_dict as cfg

# derived from https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation/
class HeatmapParser:
    """
        对输出进行后处理和解析，得到预测关键点和相应标签
        只调用 parse()函数
    """

    def __init__(self):
        self.dataset = DATASET
        self.n_joints = cfg["n_joints"]
        self.image_size = cfg["image_size"]  # 输入网络的图像的大小
        self.num_candidates = pcfg["num_candidates"]  # NMS前候选框个数=取中心点热图峰值的个数
        self.max_num_bbox = pcfg["max_num_bbox"]  # 一张图片上最多保留的目标数

        self.max_pool = torch.nn.MaxPool2d(pcfg["nms_kernel"],
                                           pcfg["nms_stride"],
                                           pcfg["nms_padding"])
        self.avg_pool = torch.nn.AvgPool2d(pcfg["region_avg_kernel"],
                                           pcfg["region_avg_stride"],
                                           pcfg["region_avg_padding"])

        self.detection_threshold = pcfg["detection_threshold"]  # 候选框检测到目标的阈值
        self.iou_threshold = pcfg["iou_threshold"]  # NMS去掉重叠框的IOU阈值
        self.tag_threshold = pcfg["tag_threshold"]
        self.use_detection_val = pcfg["use_detection_val"]
        self.ignore_too_much = pcfg["ignore_too_much"]
        self.bbox_factor = pcfg["bbox_factor"]  # 限制区域为预测框大一点点的区域，这个因子决定了放缩大小
        self.bbox_k = pcfg["bbox_k"]  # 限制区域内每张热图取得分前k个候选点。

    def nms(self, heatmaps):
        """
            热图上对每个峰值点进行nms抑制，去除掉峰值点附近的值，减少多余候选点。
        :param heatmaps:
        :return:
        """
        hm_max = self.max_pool(heatmaps)
        mask = torch.eq(hm_max, heatmaps).float()
        heatmaps *= mask
        return heatmaps

    def candidate_bbox(self, center_maps, size_maps):
        """
            根据中心点热图和宽高热图，得到k个候选框
        :param center_maps: 中心点热图： (batch, hm_size, hm_size)
        :param size_maps: 宽高热图： (batch, 2, hm_size, hm_size)， second dim = (width, height)
        :return: (batch, k, 5), last dim is (x_center, y_center, width, height, confidence)
        """
        batch, heatmap_size, _ = center_maps.shape
        candidates = torch.zeros((batch, self.num_candidates, 5), dtype=torch.float32)

        # get center points
        # center_maps = self.nms(center_maps)  # 输入前已经做了
        center_maps = center_maps.reshape((batch, -1))  # (batch, hm_size * hm_size)
        top_val, top_idx = torch.topk(center_maps, k=self.num_candidates)  # (batch, k), (batch, k)

        candidates[..., 0] = top_idx % heatmap_size  # x
        candidates[..., 1] = torch.div(top_idx, heatmap_size, rounding_mode='trunc')  # y = top_idx // heatmap_size

        # get width and height form size_maps
        size_maps = self.avg_pool(size_maps)  # 对预测的宽高热图进行大小不变的平均池化，然取宽高值只需要取中心点处的值
        for bi in range(batch):
            for ki in range(self.num_candidates):
                x, y = candidates[bi, ki, :2]
                x, y = int(x), int(y)
                candidates[bi, ki, 2] = size_maps[bi, 0, y, x]  # get region width ratio 0~1
                candidates[bi, ki, 3] = size_maps[bi, 1, y, x]  # get region height ratio 0~1
        candidates[..., 2:4] = candidates[..., 2:4].clip(0, 0.99)  # a ratio: 0~1

        candidates[..., 4] = top_val  # confidence

        # resize center x,y and size w,h from heatmap to original image
        feature_stride = self.image_size[0] / heatmap_size  # (int) default is 4
        candidates[..., :2] *= feature_stride
        candidates[..., 2:4] *= self.image_size[0]  # width and height of bbox
        return candidates

    def non_max_suppression(self, candidates):
        """
            非极大值抑制，去掉重叠率高的候选框以及得分率低的候选框
        :param candidates: 预测的bbox，(batch, k, 5), (x_center,y_center, w, h, conf), k个候选框按置信度降序排列
        :return: a list of the format: (n_images, n_boxes, 5)， 每张图片的预测框数目可能不同
                [[ [x, y, w, h, conf],  # 第一张图片的第一个预测框
                   [x, y, w, h, conf]， # 第一张图片的第二个预测框
                    ....
                    ]，
                 [ [x, y, w, h, conf],  # 第二张图片的第一个预测框
                   [x, y, w, h, conf]， # 第二张图片的第二个预测框
                    ....
                    ]，
                 ...
                ]
        """
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        time_limit = 10.0  # seconds to quit after
        t = time.time()
        output = [None] * candidates.shape[0]  # batch 张图片的预测框，非None则该图片有预测到目标框
        for i, x in enumerate(candidates):
            x = x[x[:, 4] > self.detection_threshold]  # confidence 根据obj confidence虑除背景目标
            x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # # width-height 虑除小目标

            if not x.shape[0]:
                continue  # 如果 x.shape = [0, 4]， 则当前图片没有检测到目标

            boxes = xywh2xyxy(x[:, :4])  # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            scores = x[:, 4]
            # NMS
            index = torchvision.ops.nms(boxes, scores, self.iou_threshold)
            index = index[:self.max_num_bbox]  # 一张图片最多只保留前max_num个目标信息

            output[i] = x[index].tolist()
            if (time.time() - t) > time_limit:
                break  # 超时退出

        return output

    def group_keypoints(self, bbox_list, heatmaps, tag_maps):
        """
            对每个预测框内预测关键点，
        :param bbox_list: (list) [ bbox_list_of_image1, bbox_list_of_image2, ... ], (x, y, w, h, conf) on original img
        :param heatmaps: 经过nms后的关键点预测热图
        :param tag_maps:
        :return: (list) [ keypoints_list1, keypoints_list2, ...]
        """
        batch, n_points, h, w = heatmaps.shape  # n_points = 1 central point + 21 joints, default w=h

        feature_stride = self.image_size[0] / w  # (int) default is 4
        keypoints_list = [None] * batch  # 每张图片有一个关键点列表

        for i_img, bboxes_i in enumerate(bbox_list):
            if bboxes_i is None:
                continue

            keypoints = []
            for bbox in bboxes_i:  # 图片上一个bbox预测一组关键点。
                x_center, y_center, w_bbox, h_bbox, _ = [v / feature_stride for v in bbox]  # 获取热图上的bbox
                w_bbox = int(w_bbox * self.bbox_factor)  # 比预测框的区域更大一点，防止预测框预测过小时关键点在区域外
                h_bbox = int(h_bbox * self.bbox_factor)
                x = max(0, int(x_center - w_bbox / 2 + 0.5))
                y = max(0, int(y_center - h_bbox / 2 + 0.5))

                x_bound = min(x+w_bbox, heatmaps.shape[3])
                y_bound = min(y+h_bbox, heatmaps.shape[2])
                heatmaps_region = heatmaps[i_img, :, y:y_bound, x:x_bound]  # 只是在预测框内匹配关键点。
                tag_region = tag_maps[i_img, :, y:y_bound, x:x_bound]

                heatmaps_region = heatmaps_region.reshape(n_points, -1)
                tag_region = tag_region.reshape(n_points, -1)

                print(f"{x=}\t{y=}")
                print(f"{w_bbox=}\t{h_bbox=}")
                print(f"{heatmaps_region.shape=}")

                # 取出限制区域内得分最高的前k个候选点  (n_joints, bbox_k)
                val_k, idx_k = heatmaps_region[1:].topk(k=self.bbox_k, dim=-1)  # (n_points, bbox_k), (22, 44)
                # 取出前k个候选点相应的tag标签
                tag_k = tag_region[1:].gather(dim=1, index=idx_k)  # (n_points, bbox_k)

                # 将前k个候选点的tag与中心点的tag计算L2距离，保留距离最短的候选点
                # todo 如果候选点并不是得分最高的候选点(得分高但可能是另外一只手的点)，
                #  但怎么保证只考虑tag距离能正确分组呢？ 需不需要权衡一下候选点得分。
                tag_center = tag_maps[i_img, 0, int(y_center+0.5), int(x_center+0.5)]
                abs_diff = torch.abs(tag_k - tag_center)  # 绝对差 absolute difference
                min_idx = abs_diff.argmin(dim=-1).reshape(-1, 1)

                print(f"{val_k.shape=}")
                print(f"{tag_k.shape=}")
                print(f"{abs_diff.shape=}")
                print(f"{min_idx.shape=}")
                # todo 验证正确性
                kpt_idx = idx_k.gather(dim=1, index=min_idx)  # 获取最终关键点的序号, (n_joints, 1)
                print(f"{kpt_idx.shape=}")
                conf = val_k.gather(dim=1, index=min_idx)  # 获取最终关键点的置信度, (n_joints, 1)
                tag = tag_k.gather(dim=1, index=min_idx)  # 获取最终关键点的tag, (n_joints, 1)
                x_kpt = kpt_idx % x_bound
                y_kpt = torch.div(kpt_idx, x_bound, rounding_mode='trunc')  # kpt_idx // h_bbox

                # 将关键点坐标从限制区域映射会原来的热图大小。
                print(f"{x_kpt.shape=}")
                x_kpt += x
                y_kpt += y
                print(f"{x_kpt.shape=}")
                keypoints.append(torch.cat((x_kpt, y_kpt, conf, tag), dim=-1).tolist())
            keypoints_list[i_img] = keypoints
        return keypoints_list

    def adjust_keypoints(self, keypoints, heatmaps):
        """

        :param keypoints: (list), [ k_list_1, ... k_list_batch], k_list_i = [ [x, y , conf, tag], ... ]
                            (batch, n_bbox, n_joints, joint), joint = [x, y , conf, tag]
        :param heatmaps: (batch, n_joints+1, hm_size, hm_size)  在nms前的关键点预测热图
        :return: (list) keypoints after adjustment
        """
        for batch_id, kpt_list in enumerate(keypoints):
            for bbox_id, kpt in enumerate(kpt_list):
                for joint_id, joint in enumerate(kpt):
                    x, y = joint[:2]  # column, row
                    xx, yy = int(x), int(y)
                    print(f"{xx=}\t{yy=}")
                    tmp = heatmaps[batch_id, joint_id]  # (h, w)
                    if tmp[yy, min(xx+1, tmp.shape[1]-1)] > tmp[yy, max(xx-1, 0)]:
                        x += 0.25  # 如果峰值点右侧点 > 左侧点，则最终坐标向右偏移0.25
                    else:
                        x -= 0.25

                    if tmp[min(yy+1, tmp.shape[0]-1), xx] > tmp[max(yy-1, 0), xx]:
                        y += 0.25
                    else:
                        y -= 0.25
                    keypoints[batch_id][bbox_id][joint_id][0] = x  # todo 在simpleHigherHRNet上这个是 x + 0.5
                    keypoints[batch_id][bbox_id][joint_id][1] = y  # todo 在simpleHigherHRNet上这个是 y + 0.5
        return keypoints

    def parse(self, heatmaps, size_maps, tag_maps):
        """
            解析预测热图的流程函数，返回预测结果
        ：param heatmaps: (batch, n_joints + 1, hm_size, hm_size)
        ：param size_maps: (batch, 2, hm_size, hm_size)
        ：param tag_maps: (batch, n_joints + 1, hm_size, hm_size)
        ：return:
        """
        # 1: nms 去除heatmaps各峰值领域的响应值，仅保留峰值点
        heatmaps_max = self.nms(heatmaps)

        # 2: 获取预测框bbox = （x_center, y_center, width, height）
        print(f"{heatmaps_max[:, 0].shape=}")
        bbox = self.candidate_bbox(heatmaps_max[:, 0], size_maps)
        print(f"{bbox.shape=}")
        bbox_list = self.non_max_suppression(bbox)

        # 3: Grouping keypoints by tag
        pred_keypoints = self.group_keypoints(bbox_list, heatmaps_max, tag_maps)

        # 4： 关键点优化，向次峰值方向偏移0.25
        pred_keypoints = self.adjust_keypoints(pred_keypoints, heatmaps)

        return pred_keypoints, bbox_list

