from functools import total_ordering
import time

import torch
import numpy as np
from collections import defaultdict
from utils.bbox_metric import xywh2xyxy, box_iou, bbox_iou
from utils.evaluation import count_ap
import torchvision

from config.config import DATASET, parser_cfg, config_dict as cfg


class ResultParser:
    """
        解析Centermap得到边界框，然后解析一维向量，得到关键点。
    """

    def __init__(self):
        self.max_pool = torch.nn.MaxPool2d(parser_cfg["nms_kernel"],
                                           parser_cfg["nms_stride"],
                                           parser_cfg["nms_padding"])

        self.num_candidates = parser_cfg["num_candidates"]  # NMS前候选框个数=取中心点热图峰值的个数
        self.max_num_bbox = parser_cfg["max_num_bbox"]  # 一张图片上最多保留的目标数
        self.detection_threshold = parser_cfg["detection_threshold"]  # 候选框检测到目标的阈值
        self.iou_threshold = parser_cfg["iou_threshold"]  # NMS去掉重叠框的IOU阈值
        
        self.image_size = torch.tensor(cfg['image_size'])  # (256, 256)
        self.heatmap_size = torch.tensor(cfg['hm_size'])   # (64, 64)
        # self.feature_stride = self.image_size // self.heatmap_size   # default 4
        self.feature_stride = torch.div(self.image_size, self.heatmap_size, rounding_mode='trunc')   # default 4
        self.simdr_split_ratio = cfg['simdr_split_ratio']   # default 2

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
    
    def vector_nms(self, vector):
        """给x,y向量做1d max pool

        Args:
            vector (tensor): (batch, n_joints, w or h)

        Returns:
            (tensor): (batch, n_joints, w or h)
        """
        # TODO: 这个最大池化kernel应该取多大好？ 3 or 5 ?
        v_max = torch.max_pool1d(vector, 3, 1, 1)   
        mask = torch.eq(v_max, vector).float()
        vector *= mask
        return vector

    # 关键点解析
    def get_coordinates(self, x_vectors, y_vectors, pred_bboxes):
        """获取关键点在1D 向量上的坐标"""
        # TODO: 怎么解析关键点？ 直接取框内最大值点，不考虑是否框有重合（待优化）
        # （1） 如果两个框没有相交，直接取框范围内最大得分点
        # （2） 如果框相交了，应该怎么判断？
        batch, n_joints, w = x_vectors.shape  
        _, _, h = y_vectors.shape
        
        # 预处理，抑制峰值点领域点
        x_vectors = self.vector_nms(x_vectors)
        y_vectors = self.vector_nms(y_vectors)
        
        # 在边界框内取最峰值
        coors = torch.zeros((batch, self.max_num_bbox, n_joints, 3), dtype=torch.float32).to(x_vectors.device)
        
        for i in range(batch):
            if pred_bboxes[i] is not None:
                for j, bbox in enumerate(pred_bboxes[i]):
                    bbox = np.array(bbox) * self.simdr_split_ratio
                    bbox = np.round(bbox)

                    x1, y1 = bbox[:2] - bbox[2:4]
                    x2, y2 = bbox[:2] + bbox[2:4]
                    x1, y1 = max(int(x1), 0), max(int(y1), 0)
                    x2, y2  = min(int(x2), w), min(int(y2), h)
                    
                    mask_x = torch.zeros((n_joints, w), dtype=torch.bool).to(x_vectors.device)
                    mask_x[x1: x2] = True
                    
                    mask_y = torch.zeros((n_joints, h), dtype=torch.bool).to(x_vectors.device)
                    mask_y[y1: y2] = True                    
                    
                    # todo: 取bbox范围内峰值点
                    # score_x, x = torch.topk(x_vectors[i] * mask_x, k=1)  # (n_joints, k)
                    # score_y, y = torch.topk(y_vectors[i] * mask_y, k=1)
                    score_x, x = torch.topk(x_vectors[i], k=1)  # (n_joints, k)
                    score_y, y = torch.topk(y_vectors[i], k=1)
                                      
                    x, y = x / self.simdr_split_ratio, y / self.simdr_split_ratio
                    score = (score_x + score_y) / 2
                    coors[i, j] = torch.cat([x, y, score], dim=1)
                    
        return coors
                    
    def candidate_bbox(self, center_hm, wh_hm, offset_hm):
        """
            根据中心点热图和宽高热图，得到k个候选框
        :param center_hm: 中心点热图: (batch, 1,  hm_size, hm_size)
        :param wh_hm: 宽高热图: (batch, 2, hm_size, hm_size)， second dim = (width, height)
        :param offset_hm: 中心点偏置: (tuple)
        :return: (batch, k, 5), last dim is (x_center, y_center, width, height, confidence)
        """
        batch, _, h, w = center_hm.shape
        candidates = torch.zeros((batch, self.num_candidates, 5), dtype=torch.float32).to(center_hm.device)

        # get center points
        center_hm = center_hm.reshape((batch, -1))  # (batch, hm_size * hm_size)
        top_val, top_idx = torch.topk(center_hm, k=self.num_candidates)  # (batch, k), (batch, k)

        # get x, y on center heatmap
        candidates[..., 0] = top_idx % w  # x
        candidates[..., 1] = torch.div(top_idx, w, rounding_mode='trunc')  # top_idx // w  

        # get width and height form wh heatmap and add offset into x,y
        for bi in range(batch):
            for ki in range(self.num_candidates):          
                x, y = candidates[bi, ki, :2]
                x, y = int(x), int(y)
                
                candidates[bi, ki, 2] = wh_hm[bi, 0, y, x]  # get width ratio 0~1
                candidates[bi, ki, 3] = wh_hm[bi, 1, y, x]  # get height ratio 0~1
                
                candidates[bi, ki, 0] += offset_hm[bi, 0, y, x]   # x + offset_x
                candidates[bi, ki, 1] += offset_hm[bi, 1, y, x]   # y + offset_y
                
        candidates[..., 2:4] = candidates[..., 2:4].clip(0, 0.99)  # a ratio: 0~1
        candidates[..., 4] = top_val  # confidence

        # resize center x,y and size w,h from heatmap to original image
        candidates[..., :2] *= self.feature_stride.to(center_hm.device)
        candidates[..., 2:4] *= self.image_size.to(center_hm.device)  # width and height of bbox

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

    def parse(self, centermap, pred_x=None, pred_y=None,):    
        """

        :param centermap: (batch, 5, h, w), h 和 w是热图的的宽高，默认是64
        :param pred_x:  (batch, n_joints, img_w * k), k is simdr_split_ratio
        :param pred_y:  (batch, n_joints, img_h * k)
        :return:  返回原图上的关键点坐标和预测框
                tensor: [batch, max_num_bbox, n_joints, 3] 
                list: (n_images, n_boxes, 5)
        """
        # 1： 获取原图上的边界框
        center_hm = centermap[:, 0:1]       # (batch, 1, h, w)
        wh_hm     = centermap[:, 1:3]       # (batch, 2, h, w)
        offset_hm = centermap[:, 3:]        # (batch, 2, h, w)
        
        center_hm = self.heatmap_nms(center_hm)  # 去除部分峰值
        candidates = self.candidate_bbox(center_hm, wh_hm, offset_hm)
        pred_bboxes = self.non_max_suppression(candidates)  # list

        # 2： 获取关键点
        pred_kpts = self.get_coordinates(pred_x, pred_y, pred_bboxes)

        return pred_kpts, pred_bboxes

    @staticmethod
    def evaluate_ap(pred_bboxes, gt_bboxes, iou_thr=None):
        gt_bboxes = gt_bboxes.tolist() if isinstance(gt_bboxes, torch.Tensor) else gt_bboxes
        ap50, ap = count_ap(pred_boxes=pred_bboxes, gt_boxes=gt_bboxes, iou_threshold=iou_thr)
        # print(pred_bboxes[0])
        # print(gt_bboxes[0])

        return ap50, ap

    def evaluate_pck(self, pred_kpts, gt_kpts, thr=0.2):
        """计算多手PCK， 
        1、先看预测的目标点中心点与真值点中心点的距离来匹配识别目标。
        2、分别对识别出的目标计算PCK
        为了简化代码，将关键点的格式标准化，第二维固定为数据集中最大出现的目标个数，而不是实际目标数。
        Args:
            pred_kpts (tensor): 预测的关键点  [batch, max_num_bbox, n_joints, 3], (x, y, score)
            gt_kpts (tensor): 真值关键点  [batch, max_num_bbox, n_joints, 3], (x, y, vis)
        """
        def get_center(kpts):
                # kpts.shape = (max_num_bbox, n_joints, 3)
                kpts = kpts[torch.sum(kpts[:,:, 2] != 0, dim=-1) > 0]  # 保留实际手部个数的关键点
                num_object = kpts.shape[0]      
                num_vis_joints = torch.sum(kpts[:,:,2] > 0, dim=1)[:, None]  # (num_object, 1)      
                center_xy = kpts[:,:,:2].sum(dim=1) / num_vis_joints  # (num_object, 2)
                return center_xy, num_object, num_vis_joints
        
        def get_pck(pred, gt):
            # 只计算可见点
            gt_vis = gt[gt[:,2] > 0]
            pred_vis = pred[gt[:,2] > 0]
            
            # 计算bbox的w,h
            (x1, y1), _ = gt_vis[:, :2].min(dim=0)
            (x2, y2), _ = gt_vis[:, :2].max(dim=0)
            w, h = x2 - x1, y2 - y1
            
            pck = torch.sum(torch.norm(gt_vis-pred_vis,p=2, dim=1) / max(w, h) < thr) / gt_vis.shape[0]
            return pck.item()
            
        pck_list = []
        for _pred_kpts, _gt_kpts in zip(pred_kpts, gt_kpts):
            center_gt, num_gt, num_vis_joints_gt = get_center(_gt_kpts)
            center_pred, num_pred, num_vis_joints_pred = get_center(_pred_kpts)
            
            # 去除冗余占位部分
            _gt_kpts = _gt_kpts[:num_gt]    
            _pred_kpts = _pred_kpts[:num_pred]
            
            for center, pred in zip(center_pred, _pred_kpts) :
                distance = torch.pow(center_gt - center, 2).sum(dim=1)
                min_idx = distance.argmin()
                gt = _gt_kpts[min_idx]
                
                pck = get_pck(pred, gt)
                pck_list.append(pck)
        
        avg_pck = sum(pck_list) / len(pck_list) if len(pck_list) != 0 else 0
        return avg_pck
            
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

    parser = ResultParser()
    k, b = parser.parse(kpt_hm, c_hm, s_hm, (256, 256))
    print(f"{k=}")
    print(f"{b=}")
    gt_b = [[[12.0, 12.0, 100.44000244140625, 100.44000244140625, 1.0]],
            [[13.0, 10.0, 253.44000244140625, 253.44000244140625, 1.0]]]
    a50, a = parser.evaluate_ap(b, gt_b)
    print(f"{a50=}")
    print(f"{a=}")
