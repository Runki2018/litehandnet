import numpy as np
import torch
from utils.post_processing.evaluation.top_down_eval import (keypoints_from_heatmaps,
                                                            keypoints_from_simdr)

def _TensorToNumpy(tensor):
    return tensor.cpu().detach().numpy()

class TopDownDecoder:
    """
        对模型输出进行解码,得到预测结果
    """
    def __init__(self, cfg):
        self.image_size = np.array(cfg.DATASET.image_size)
        self.heatmap_size = np.array(cfg.DATASET.heatmap_size)
        self.num_joints = cfg.DATASET.num_joints
        if cfg.PIPELINE.unbiased_encoding:   
            self.post_process = 'unbiased'
        else:
            self.post_process = 'default'
        self.kernel = cfg.PIPELINE.kernel[0]
        self.use_udp = cfg.PIPELINE.use_udp
        self.k = cfg.PIPELINE.get('simdr_split_ratio', 0)


    def decode(self, meta, model_output):
        """解码

        Args:
            meta (dict): 包含当前batch的image、target、annotations信息
            model_output (list or Tensor): [N, K, H, W],模型输出
        """

        score = _TensorToNumpy(meta['bbox_score'])
        bbox_ids = _TensorToNumpy(meta['bbox_id'])
        output_heatmap = _TensorToNumpy(model_output[:, :self.num_joints])
        image_paths = meta['image_file']
        center = _TensorToNumpy(meta['center'])
        scale = _TensorToNumpy(meta['scale'])  # (W, H) / 200
        
        preds, maxvals = keypoints_from_heatmaps(
            heatmaps=output_heatmap,
            center=center,
            scale=scale,
            post_process=self.post_process,  # None, 'default', 'unbiased'
            kernel=self.kernel,              # kernel大小与sigma必须匹配
            use_udp=self.use_udp,
            target_type='GaussianHeatmap')
        
        batch_size = model_output.shape[0]
        all_preds = np.zeros((batch_size, self.num_joints, 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)

        all_preds[:, :, 0:2] = preds[:, :, 0:2]   # 原图关键点坐标
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = center[:, 0:2]   # bbox的 中心点
        all_boxes[:, 2:4] = scale[:, 0:2]   # bbox的 宽高/200
        all_boxes[:, 4] = np.prod(scale * 200.0, axis=1)   # bbox的面积
        all_boxes[:, 5] = score   # bbox的得分

        # for i in range(batch_size):
        result = {}
        result['preds'] = all_preds   # 预测出的原图关键点坐标
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids.tolist()
        result['output_heatmap'] = output_heatmap

        return result
    
    def decode_simdr(self, meta, model_output):
        """Simdr解码

        Args:
            meta (dict): 包含当前batch的image、target、annotations信息
            model_output (list or Tensor): [N, K, H, W],模型输出
        """

        score = _TensorToNumpy(meta['bbox_score'])
        bbox_ids = _TensorToNumpy(meta['bbox_id'])
        output_heatmap = _TensorToNumpy(model_output[:, :self.num_joints])
        simdr_x = _TensorToNumpy(meta['simdr_x'])   # [B, K, Img_W*k]
        simdr_y = _TensorToNumpy(meta['simdr_y'])   # [B, K, Img_H*k]
        image_paths = meta['image_file']
        center = _TensorToNumpy(meta['center'])
        scale = _TensorToNumpy(meta['scale'])  # (W, H) / 200

        batch_size = simdr_x.shape[0]
        # all_preds = self._get_coordinates_from_vectors(simdr_x, simdr_y)
        all_preds = keypoints_from_simdr(simdr_x, simdr_y, center, scale, self.k)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)

        all_boxes[:, 0:2] = center[:, 0:2]   # bbox的 中心点
        all_boxes[:, 2:4] = scale[:, 0:2]   # bbox的 宽高/200
        all_boxes[:, 4] = np.prod(scale * 200.0, axis=1)   # bbox的面积
        all_boxes[:, 5] = score   # bbox的得分

        # for i in range(batch_size):
        result = {}
        result['preds'] = all_preds   # 预测出的原图关键点坐标
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids.tolist()
        result['output_heatmap'] = output_heatmap

        return result

    def _get_coordinates_from_vectors(self, x_vectors, y_vectors, k=2):
        """获取关键点在1D 向量上的坐标"""
        batch, n_joints = x_vectors.shape[:2]
        coors = np.zeros((batch, n_joints, 3), dtype=np.float32)
        
        for i in range(batch):
            x_max_idx = x_vectors[i].argmax(axis=1).reshape((-1, 1))
            x_max_val = x_vectors[i].max(axis=1).reshape((-1, 1))

            y_max_idx = y_vectors[i].argmax(axis=1).reshape((-1, 1))
            y_max_val = y_vectors[i].max(axis=1).reshape((-1, 1))

            x, y = x_max_idx / self.k, y_max_idx / self.k
            score = (x_max_val + y_max_val) / 2
            coors[i] = np.concatenate([x, y, score], axis=1)

        return coors

