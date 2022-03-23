import time
import torch
import numpy as np
from utils.bbox_metric import xywh2xyxy, box_iou, bbox_iou
from utils.evaluation import count_ap
from utils.heatmap_post_processing import adjust_keypoints_by_DARK, adjust_keypoints_by_offset
import torchvision

from config import pcfg

def _fdiv(a, b, rounding_mode='trunc'):
    return  torch.div(a, b, rounding_mode=rounding_mode)

class ResultParser:
    """
        解析Centermap得到边界框，然后解析一维向量，得到关键点。
    """

    def __init__(self, cfg):
        self.cfg = cfg
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
        
        self.bbox_factor = pcfg["bbox_factor"]
        self.image_size = torch.tensor(cfg['image_size'])  # (256, 256)
        self.image_area = cfg['image_size'][0] * cfg['image_size'][1]
        if cfg['model'] == 'srhandnet':  
            self.heatmap_size = torch.tensor([cfg['hm_size'][-1],
                                              cfg['hm_size'][-1]])   # (64, 64)
        else:
            self.heatmap_size = torch.tensor(cfg['hm_size'])

        self.feature_stride = _fdiv(self.image_size, self.heatmap_size)   # default 4
        self.simdr_split_ratio = cfg['simdr_split_ratio']   # default 2
        self.bbox_alpha = cfg['bbox_alpha']
        self.cd_enabled = cfg['with_region_map']
        self.cd_reduction = cfg['cycle_detection_reduction']

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

    def get_coordinates_from_heatmaps(self, heatmaps):
        """通过取峰值点获取关键点在热图上的坐标""" 
        batch, n_joints, h, w = heatmaps.shape
        try:    
            top_val, top_idx = torch.topk(heatmaps.reshape((batch, n_joints, -1)), k=1)
        except RuntimeError:
            print(f"\n\n{heatmaps.shape=}\n\n")

        kpts = torch.zeros((batch, n_joints, 3), dtype=torch.float32).to(heatmaps.device)
        kpts[..., 0] = (top_idx % w).reshape((batch, n_joints))  # x
        kpts[..., 1] = _fdiv(top_idx, w).reshape((batch, n_joints))  # y
        # batch_kpts[..., 1] = (top_idx // w).reshape((batch, n_joints))  # y
        kpts[..., 2] = top_val.reshape((batch, n_joints))  # c: score

        return kpts

    # 关键点解析
    def get_coordinates_from_vectors(self, x_vectors, y_vectors, pred_bboxes):
        """获取关键点在1D 向量上的坐标"""
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

                    x1, y1 = bbox[:2] - bbox[2:4] / 2
                    x2, y2 = bbox[:2] + bbox[2:4] / 2
                    x1, y1 = max(int(x1), 0), max(int(y1), 0)
                    x2, y2  = min(int(x2), w), min(int(y2), h)
                    
                    mask_x = torch.zeros((n_joints, w), dtype=torch.bool).to(x_vectors.device)
                    mask_x[:, x1: x2] = True
                    
                    mask_y = torch.zeros((n_joints, h), dtype=torch.bool).to(x_vectors.device)
                    mask_y[:, y1: y2] = True        
  
                    score_x, x = torch.topk(x_vectors[i] * mask_x, k=1)  # (n_joints, k)
                    score_y, y = torch.topk(y_vectors[i] * mask_y, k=1)
                                      
                    x, y = x / self.simdr_split_ratio, y / self.simdr_split_ratio
                    score = (score_x + score_y) / 2
                    coors[i, j] = torch.cat([x, y, score], dim=1)
                    
        return coors
    
    def candidate_bbox(self, center_maps, size_maps):
        """
            根据中心点热图和宽高热图，得到k个候选框
        :param center_maps: 中心点热图： (batch, 1,  hm_size, hm_size)
        :param size_maps: 宽高热图： (batch, 2, hm_size, hm_size)， second dim = (width, height)
        :return: (batch, k, 5), last dim is (x_center, y_center, width, height, confidence)
        """
        device = center_maps.device
        batch, _, h, w = center_maps.shape
        candidates = torch.zeros((batch, self.num_candidates, 5), dtype=torch.float32)

        # get center points
        top_val, top_idx = torch.topk(center_maps.reshape((batch, -1)), k=self.num_candidates)  # (batch, k), (batch, k)

        candidates[..., 0] = top_idx % w  # x
        candidates[..., 1] = _fdiv(top_idx, w)  # top_idx // w

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
        
        for i in range(self.num_candidates):
            kpt = candidates[:, i:i+1, :2]
            if self.cfg['DARK']:
                kpt = adjust_keypoints_by_DARK(kpt, center_maps)
            else:
                adjust_keypoints_by_offset(kpt, center_maps)
            candidates[:, i:i+1,:2] = torch.from_numpy(kpt)

        # resize center x,y and size w,h from heatmap to original image
        candidates[..., :2] *= self.feature_stride
        candidates[..., 2:4] *= self.image_size  # width and height of bbox             
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
            # confidence 根据obj confidence虑除背景目标
            x = x[x[:, 4] > self.detection_threshold]  
            # width-height 虑除小目标
            x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  

            if not x.shape[0]:
                continue  # 如果 x.shape = [0, 4]， 则当前图片没有检测到目标
                
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            boxes = xywh2xyxy(x[:, :4])  
            scores = x[:, 4]
            # NMS
            index = torchvision.ops.nms(boxes, scores, self.iou_threshold)
            index = index[:self.max_num_bbox]  # 一张图片最多只保留前max_num个目标信息

            output[i] = x[index].tolist()
            if (time.time() - t) > time_limit:
                break  # 超时退出

        return output

    def get_pred_bbox(self, region_map):
        """从 Region map获取bbox
        Args:
            region_map (tensor): [batch, 3, h, w]

        Returns:
            list: [batch, n_bbox, 5]
        """
        center_hm = region_map[:, 0:1]        # (batch, 1, h, w)
        size_hm = region_map[:, 1:3]         # (batch, 2, h, w)
        center_hm = self.heatmap_nms(center_hm)  # 去除部分峰值
        candidates = self.candidate_bbox(center_hm, size_hm)
        pred_bboxes = self.non_max_suppression(candidates)  # list
        return pred_bboxes
    
    def get_pred_kpt(self, heatmap, resized=False):
        """ 获取单手图片的关键点
        Args:
            heatmap (tensor): [batch, n_joints, H, W]
        Returns:
            tensor: (batch, n_joints, 3)
        """
        hm2kpt = self.get_coordinates_from_heatmaps(heatmap)
        device = hm2kpt.device
        if self.cfg['DARK']:
            hm2kpt = adjust_keypoints_by_DARK(hm2kpt.clone(), heatmap)
        else:
            hm2kpt = adjust_keypoints_by_offset(hm2kpt.clone(), heatmap)
            
        hm2kpt = torch.as_tensor(hm2kpt, device=device)
        
        if resized:
            hm2kpt[..., :2] *= self.feature_stride.to(hm2kpt.device)
        return hm2kpt
    
    def get_group_keypoints(self, model, img, bbox_list, heatmaps):
        """
            对每个预测框内预测关键点
        :param bbox_list: (list) [ bbox_list_of_image1, bbox_list_of_image2, ... ], (x, y, w, h, conf) on original img
        :param heatmaps (tensor): [batch, n_joints, H, W]  经过nms后的关键点预测热图
        :return: (tensor) (batch, self.max_num_bbox, n_joints, 3)
        """
        batch, n_joints, h, w = heatmaps.shape  
        pred_kpts = torch.zeros((batch, self.max_num_bbox, n_joints, 3))

        for img_idx, bboxes in enumerate(bbox_list):
            if bboxes is None:
                continue     
            for bbox_idx, bbox in enumerate(bboxes):  # 图片上一个bbox预测一组关键点。
                if self.cd_enabled and \
                    self._is_cycle_detection( 
                        bbox, bboxes, pcfg['cd_iou'], pcfg['cd_ratio']):
                        
                    pred_kpts[img_idx, bbox_idx] = \
                        self._get_second_result(model, img, bbox, heatmaps, img_idx)
                else:
                    pred_kpts[img_idx, bbox_idx] = \
                        self._get_first_result(bbox, heatmaps, img_idx)
        return pred_kpts
    
    def _is_cycle_detection(self, bbox, bboxes, iou_thr=0.3, ratio=0.1):
        """判断是否需要循环检测： 1、有重合 2、手部区域面积占原图比率小

        Args:
            bbox (list): [x_center, y_center, w, h]
            bboxes (list): [bbox1, bbox2, ...]
            ratio (float, optional):  Defaults to 0.1.
        """
        area = bbox[2] * bbox[3]
        r = area / self.image_area
        if r <= ratio and area != 0:
            # print(f"less than {ratio=}")
            return True
        
        iou = bbox_iou(bbox, bboxes, x1y1x2y2=False, DIoU=True)
        if sum(iou > iou_thr) > 1:
            return True

        return False
    
    def _get_first_result(self, bbox, heatmaps, img_idx:int):
        """首次检测结果"""
        feature_stride = self.feature_stride[0]  # (int) default is 4
        x_center, y_center, w_bbox, h_bbox = [v / feature_stride for v in bbox][:4]

        # 比预测框的区域更大一点，防止预测框预测过小时关键点在区域外
        w_bbox = int(w_bbox * self.bbox_factor) 
        h_bbox = int(h_bbox * self.bbox_factor)
        # 获取预测框的左上角点和右下角点
        ul_x = max(0, int(x_center - w_bbox / 2 + 0.5))
        ul_y = max(0, int(y_center - h_bbox / 2 + 0.5))
        br_x = min(ul_x + w_bbox, heatmaps.shape[3])
        br_y = min(ul_y + h_bbox, heatmaps.shape[2])
        
        # 只是在预测框内匹配关键点。
        part_heatmaps = heatmaps[img_idx:img_idx+1, :, ul_y:br_y, ul_x:br_x]       
        if 0 in part_heatmaps.shape:
            ul_x, ul_y = 0, 0
            part_heatmaps = heatmaps[img_idx:img_idx+1] 
        kpt = self.get_pred_kpt(part_heatmaps)  # (1, n_joints, 3)

        # 将关键点坐标从限制区域映射会原来的热图大小。
        kpt[:, :, :2] += torch.tensor([ul_x, ul_y], device=kpt.device)
        kpt[:, :, :2] *= self.feature_stride.to(kpt.device)
        return kpt
  
    def _get_second_result(self, model, img, bbox, heatmaps, img_idx:int):
        """循环检测得到二次结果"""
        x, y, w, h = bbox[:4]
        if w * h == 0:
            return self._get_first_result(heatmaps, img_idx)

        # ? 放大预测框，尽量可能包含手部
        w, h = w * self.bbox_factor, h * self.bbox_factor
        W, H = self.image_size
        x1, y1 = max(0, int(x - w/2 + 0.5)) , max(0, int(y - h/2 + 0.5)) 
        x2, y2 = min(W, int(x + w/2 + 0.5)) , min(H, int(y + h/2 + 0.5)) 
        w, h = x2 - x1, y2 - y1
        img_crop = img[img_idx:img_idx+1, :, y1:y2, x1:x2]  # (1, 3, w, h)
        
        size = _fdiv(H, self.cd_reduction), _fdiv(W, self.cd_reduction)
        # mode 默认为nearest， 其他modes: linear | bilinear | bicubic | trilinear
        img_crop = torch.nn.functional.interpolate(img_crop, size=size) 
        if self.simdr_split_ratio > 0:
            hm_list, _, _ = model(img_crop)
        else:
            hm_list = model(img_crop)
            
        kpt = self.get_pred_kpt(hm_list[-1][:, :-3])  # (1, n_joints, 3)
        kpt[:, :, :2] *= self.feature_stride.to(kpt.device)
        kpt[:, :, :2] *= torch.tensor([w / size[1], h / size[0]], device=kpt.device)
        kpt[:, :, :2] += torch.tensor([x1, y1], device=kpt.device)
        return kpt
        
    @staticmethod
    def evaluate_ap(pred_bboxes, gt_bboxes, iou_thr=None):
        gt_bboxes = gt_bboxes.tolist() if isinstance(gt_bboxes, torch.Tensor) else gt_bboxes
        ap50, ap = count_ap(pred_boxes=pred_bboxes, gt_boxes=gt_bboxes, iou_threshold=iou_thr)
        return ap50, ap

    def evaluate_pck(self, pred_kpts, gt_kpts, bboxes, thr=0.2):
        """计算多手PCK， 
        1、先看预测的目标点中心点与真值点中心点的距离来匹配识别目标。
        2、分别对识别出的目标计算PCK
        为了简化代码，将关键点的格式标准化，第二维固定为数据集中最大出现的目标个数，而不是实际目标数。
        Args:
            pred_kpts (tensor): 预测的关键点  [batch, max_num_bbox, n_joints, 3], (x, y, score)
            gt_kpts (tensor): 真值关键点  [batch, max_num_bbox, n_joints, 3], (x, y, vis)
            bboxes (tensor): 真值关键点  [batch, n_hand, 4], (cx, cy, w, h)
        """
        def get_center(kpts):
                # kpts.shape = (max_num_bbox, n_joints, 3)
                kpts = kpts[torch.sum(kpts[:,:, 2] > 0, dim=-1) > 0]  # 保留实际手部个数的关键点
                num_object = kpts.shape[0]      
                num_vis_joints = torch.sum(kpts[:,:,2] > 0, dim=1)[:, None]  # (num_object, 1)      
                center_xy = kpts[:,:,:2].sum(dim=1) / num_vis_joints  # (num_object, 2)
                return center_xy, num_object, num_vis_joints
        
        def get_pck(pred, gt, wh):
            # 只计算可见点
            gt_vis = gt[gt[:,2] > 0]
            pred_vis = pred[gt[:,2] > 0]

            # 计算bbox的w,h      
            pck = torch.sum(torch.norm(gt_vis-pred_vis,p=2, dim=1) / torch.max(wh) < thr) / gt_vis.shape[0]
            return pck.item()
            
        pck_list = []
        bboxes = bboxes.to(pred_kpts.device)
        gt_kpts= gt_kpts.to(pred_kpts.device)
        for _pred_kpts, _gt_kpts, bbox in zip(pred_kpts, gt_kpts, bboxes):
            center_pred, num_pred, num_vis_joints_pred = get_center(_pred_kpts)        
            _pred_kpts = _pred_kpts[:num_pred]  # 去除冗余占位部分
            
            for center, pred in zip(center_pred, _pred_kpts):
                distance = torch.pow(bbox[:, :2] - center, 2).sum(dim=1)
                min_idx = distance.argmin()
                gt = _gt_kpts[min_idx]
                
                pck = get_pck(pred, gt, bbox[min_idx, :2])
                pck_list.append(pck)
        
        avg_pck = sum(pck_list) / len(pck_list) if len(pck_list) != 0 else 0
        return avg_pck
