import time

import torch
from collections import defaultdict
from utils.bbox_metric import xywh2xyxy, box_iou, bbox_iou
from utils.evaluation import count_ap
import torchvision

from config.config import DATASET, pcfg, config_dict as cfg


class HeatmapParser_SH:
    """
        解析单手姿态估计中输出的热图，得到原图上的关键点和边界框
    """

    def __init__(self):
        kernel_size = pcfg["region_avg_kernel"]
        self.avg_pool = torch.nn.AvgPool2d(kernel_size,
                                           pcfg["region_avg_stride"],
                                           (kernel_size - 1) // 2)

        self.max_pool = torch.nn.MaxPool2d(pcfg["nms_kernel"],
                                           pcfg["nms_stride"],
                                           pcfg["nms_padding"])

        self.num_candidates = pcfg["num_candidates"]  # NMS前候选框个数=取中心点热图峰值的个数
        self.max_num_bbox = pcfg["max_num_bbox"]  # 一张图片上最多保留的目标数
        self.detection_threshold = pcfg["detection_threshold"]  # 候选框检测到目标的阈值
        self.iou_threshold = pcfg["iou_threshold"]  # NMS去掉重叠框的IOU阈值

    def heatmap_nms(self, heatmaps):
        """
            热图上对每个峰值点进行nms抑制，去除掉峰值点附近的值，减少多余候选点。
        :param heatmaps:
        :return:
        """
        hm_max = self.max_pool(heatmaps)
        mask = torch.eq(hm_max, heatmaps).float()
        heatmaps *= mask
        return heatmaps

    # 关键点解析
    @staticmethod
    def get_coordinates(heatmaps):
        """获取关键点在热图上的坐标"""
        batch, n_joints, h, w = heatmaps.shape
        top_val, top_idx = torch.topk(heatmaps.reshape((batch, n_joints, -1)), k=1)

        batch_kpts = torch.zeros((batch, n_joints, 3))
        batch_kpts[..., 0] = (top_idx % w).reshape((batch, n_joints))  # x
        # batch_kpts[..., 1] = torch.div(top_idx, w, rounding_mode='floor').reshape((batch, n_joints))  # y
        batch_kpts[..., 1] = (top_idx // w).reshape((batch, n_joints))  # y
        batch_kpts[..., 2] = top_val.reshape((batch, n_joints))  # c: score

        return batch_kpts

    def candidate_bbox(self, center_maps, size_maps, image_size=(352, 352)):
        """
            根据中心点热图和宽高热图，得到k个候选框
        :param center_maps: 中心点热图： (batch, 1,  hm_size, hm_size)
        :param size_maps: 宽高热图： (batch, 2, hm_size, hm_size)， second dim = (width, height)
        :param image_size: 输入图像的大小： (tuple)
        :return: (batch, k, 5), last dim is (x_center, y_center, width, height, confidence)
        """
        batch, _, h, w = center_maps.shape
        # print(f"{center_maps.shape=}")
        candidates = torch.zeros((batch, self.num_candidates, 5), dtype=torch.float32)

        # get center points
        # center_maps = self.nms(center_maps)  # 输入前已经做了
        center_maps = center_maps.reshape((batch, -1))  # (batch, hm_size * hm_size)
        top_val, top_idx = torch.topk(center_maps, k=self.num_candidates)  # (batch, k), (batch, k)

        candidates[..., 0] = top_idx % w  # x
        candidates[..., 1] = top_idx // w  # torch.div(top_idx, w, rounding_mode='trunc')
        # print(f"{center_maps.shape=}")
        # print(f"{top_idx=}")
        # print(f"{candidates=}")

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
        image_size = torch.tensor(image_size) if not isinstance(image_size, torch.Tensor) else image_size
        heatmap_size = torch.tensor([w, h])
        feature_stride = image_size / heatmap_size  # (int) default is 4
        candidates[..., :2] *= feature_stride
        candidates[..., 2:4] *= image_size  # width and height of bbox
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

    @staticmethod
    def adjust_keypoints(keypoints, heatmaps):
        """

        :param keypoints: (batch, n_joints, 3), last dim = [x, y , conf]
        :param heatmaps: (batch, n_joints, hm_size, hm_size)  在nms前的关键点预测热图
        :return: (list) keypoints after adjustment
        """
        batch, n_joints, _ = keypoints.shape
        keypoints = keypoints.detach()
        for batch_id in range(batch):
            for joint_id in range(n_joints):
                x, y = keypoints[batch_id, joint_id, :2]  # column, row
                xx, yy = int(x), int(y)
                # print(f"{xx=}\t{yy=}")
                tmp = heatmaps[batch_id, joint_id]  # (h, w)
                if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                    x += 0.25  # 如果峰值点右侧点 > 左侧点，则最终坐标向右偏移0.25
                else:
                    x -= 0.25

                if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(yy - 1, 0), xx]:
                    y += 0.25
                else:
                    y -= 0.25
                keypoints[batch_id, joint_id, 0] = x  # todo 在simpleHigherHRNet上这个是 x + 0.5
                keypoints[batch_id, joint_id, 1] = y  # todo 在simpleHigherHRNet上这个是 y + 0.5
        return keypoints

    def parse(self, heatmaps,
                    center_maps=None,
                    size_maps=None,
                    image_size=(256, 256), scale_factor=1):
        """

        :param heatmaps: (batch, n_joints, h, w)
        :param center_maps:  (batch, 1, h, w)
        :param size_maps: (batch, 2, h, w)
        :param image_size: (tuple)
        :param scale_factor: float, default 1.25, scale factor of side
        :return:  返回原图上的关键点坐标和预测框
        """
        image_size = torch.tensor(image_size)
        heatmap_size = torch.tensor([heatmaps.shape[3], heatmaps.shape[2]])  # w, h

        # 1： 获取原图上的边界框
        if center_maps is None or size_maps is None:
            pred_bboxes = None
        else:
            center_maps = self.heatmap_nms(center_maps)
            candidates = self.candidate_bbox(center_maps, size_maps, image_size)
            pred_bboxes = self.non_max_suppression(candidates)  # list
            # # 人为调节宽高
            # for bboxes in pred_bboxes:
            #     if bboxes is not None:
            #         for bbox in bboxes:
            #             print(f"{bbox=}")
            #             bbox[2] *= scale_factor
            #             bbox[3] *= scale_factor

        # 2： 获取关键点
        kpt = self.get_coordinates(heatmaps)  # 0 center point, 1: keypoints
        kpt = self.adjust_keypoints(kpt.clone(), heatmaps)
        kpt[:, :, :2] *= image_size / heatmap_size  # (batch, n_joints, 3)


        return kpt, pred_bboxes

    @staticmethod
    def evaluate_ap(pred_bboxes, gt_bboxes, iou_thr=None):
        gt_bboxes = gt_bboxes.tolist() if isinstance(gt_bboxes, torch.Tensor) else gt_bboxes
        ap50, ap = count_ap(pred_boxes=pred_bboxes, gt_boxes=gt_bboxes, iou_threshold=iou_thr)
        print(pred_bboxes[0])
        print(gt_bboxes[0])

        return ap50, ap

# ----------------------------------------------


if __name__ == '__main__':
    kpt_hm = torch.zeros((2, 4, 64, 64))
    kpt_hm[..., 3, 3] = 1
    kpt_hm[..., 3, 2] = 0.5
    kpt_hm[..., 2, 3] = 0.5

    c_hm = torch.zeros(2, 1, 64, 64)
    c_hm[..., 3, 3] = 1
    s_hm = torch.zeros(2, 2, 64, 64)
    s_hm[..., 0:7, 0:7] = 1.
    print(f"{s_hm=}")

    parser = HeatmapParser_SH()
    k, b = parser.parse(kpt_hm, c_hm, s_hm, (256, 256))
    print(f"{k=}")
    print(f"{b=}")
    gt_b = [[[12.0, 12.0, 100.44000244140625, 100.44000244140625, 1.0]],
            [[13.0, 10.0, 253.44000244140625, 253.44000244140625, 1.0]]]
    a50, a = parser.evaluate_ap(b, gt_b)
    print(f"{a50=}")
    print(f"{a=}")
