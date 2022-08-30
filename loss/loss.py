import torch
from torch import nn
import torch.nn.functional as F
from . import *


class SRHandNetLoss(nn.Module):
    """less is more"""
    def __init__(self, cfg):
        super().__init__()
        out_c = cfg.MODEL.get('output_channel', 24)
        pred_bbox = cfg.MODEL.get('pred_bbox', False)
        self.mse_loss = DistanceLoss(loss_type='L2', reduction='mean')
        if pred_bbox and out_c == 24:  # with region map
            self.smoothl1_loss = DistanceLoss(reduction='mean')
        else:
            self.smoothl1_loss = None
        self.num_out = 4
        self.loss_weight = cfg.LOSS.loss_weight
        assert len(self.loss_weight) == self.num_out

    def forward(self, outputs, meta):
        """
        Args:
            outputs (list[Tensor]): [tensor[N, K, H, W], ...]
            targets (list[Tensor]): [tensor[N, K, H, W], ...]
            target_weight (list[Tensor] or Tensor): [tensor[N, K, 1], ...]

        Returns:
            LOSS: scalar 
        """
        target = meta['target']
        target_weight = meta['target_weight']
        if self.smoothl1_loss != None:
            loss, loss_dict = self._forward_with_regionmap(outputs, target, target_weight)
        else:
            loss, loss_dict = self._forward_only_heatmap(outputs, target, target_weight)
        return loss, loss_dict

    def _forward_with_regionmap(self, outputs, targets, target_weight):
        kpt_loss, wh_loss = 0, 0
        device = outputs[-1].device
        for i in range(self.num_out):
            pred_point, pred_wh = torch.split(outputs[i], (22, 2), dim=1)
            gt_point, gt_wh = torch.split(targets[i], (22, 2), dim=1)
            if isinstance(target_weight, list):
                weight_point, weight_wh =  torch.split(target_weight[i], (22, 2), dim=1)
            else:
                weight_point, weight_wh = torch.split(target_weight, (22, 2), dim=1)

            kpt_loss += self.mse_loss(pred_point, gt_point.to(device),
                                      weight_point.to(device)) * self.loss_weight[i]
            wh_loss += self.smoothl1_loss(pred_wh, gt_wh.to(device),
                                          weight_wh.to(device)) * self.loss_weight[i]
        loss = kpt_loss + wh_loss
        loss_dict = dict(kpt_loss=kpt_loss.item(), wh_loss=wh_loss.item())
        return loss, loss_dict

    def _forward_only_heatmap(self, outputs, targets, target_weight):
        loss = 0
        device = outputs[-1].device
        for i in range(self.num_out):
            loss += self.mse_loss(outputs[i], targets[i].to(device), 
                                      target_weight[i].to(device)) * self.loss_weight[i]
        loss_dict = dict(kpt_loss=loss.item())
        return loss, loss_dict


class TopdownHeatmapLoss(nn.Module):
    """
        MTL多任务学习,自动权重调节: https://zhuanlan.zhihu.com/p/367881339
    """

    def __init__(self, cfg):
        super().__init__()
        loss_type = cfg.LOSS.get('dl_type', 'L2')
        balance = cfg.MODEL.name != 'atthandnet'
        self.heatmap_loss = DistanceLoss(loss_type=loss_type, reduction='mean', balance=balance)
        # self.heatmap_loss = DistanceLoss(loss_type='SmoothL1Loss', reduction='mean')
        if cfg.PIPELINE.simdr_split_ratio > 0:
             # TODO:将这个参数也放入优化器的参数优化列表中
            self.simdr_loss = SimDRLoss(cfg)
        else:
            self.simdr_loss = None

        self.loss_weight = cfg.LOSS.loss_weight
        self.auto_weight = cfg.LOSS.auto_weight
        if self.auto_weight:
            params = torch.ones(len(self.loss_weight), requires_grad=True)
            # TODO:将这个参数也放入优化器的参数优化列表中
            self.p = nn.Parameter(params, requires_grad=True)

    def forward(self, output, meta):
        loss_dict = {}
        device = output.device
        
        target = meta['target'].to(device)
        target_weight = meta['target_weight'].to(device)
        
        # print(f"{target.shape=}\t{output.shape=}")
        loss_dict['heatmap'] = self.loss_weight[0] * \
            self.heatmap_loss(output, target, target_weight)

        if self.simdr_loss != None:
            simdr_x = meta['simdr_x'].to(device)
            simdr_y = meta['simdr_y'].to(device)
            loss_dict['simdr'] = self.loss_weight[1] * \
                self.simdr_loss(output, simdr_x, simdr_y, target_weight)

        loss = 0
        for k, v in loss_dict.items():
            loss += v
            loss_dict[k] = v.item()
        return loss, loss_dict


