import torch
from torch import nn
import torch.nn.functional as F
from . import *


class SRHandNetLoss(nn.Module):
    """less is more"""
    def __init__(self, cfg):
        super().__init__()
        out_c = cfg.MODEL.get('output_channel', 24)
        self.mse_loss = JointsDistanceLoss(use_target_weight=True, loss_type='MSE')
        if out_c == 24:  # with region map
            self.smoothl1_loss = JointsDistanceLoss(use_target_weight=True,
                                                    loss_type='smoothl1')
        else:
            self.smoothl1_loss = None
        self.num_out = 4
        self.loss_weight = cfg.LOSS.loss_weight
        assert len(self.loss_weight) == self.num_out

    def forward(self, outputs, targets, target_weight):
        """

        Args:
            outputs (list[Tensor]): [tensor[N, K, H, W], ...]
            targets (list[Tensor]): [tensor[N, K, H, W], ...]
            target_weight (list[Tensor] or Tensor): [tensor[N, K, 1], ...]

        Returns:
            LOSS: scalar 
        """
        if self.smoothl1_loss != None:
            loss, loss_dict = self._forward_with_regionmap(outputs, targets, target_weight)
        else:
            loss, loss_dict = self._forward_only_heatmap(outputs, targets, target_weight)
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

            kpt_loss += self.mse_loss(pred_point,
                                      gt_point.to(device),
                                      weight_point.to(device)) * self.loss_weight[i]
            wh_loss += self.smoothl1_loss(pred_wh,
                                          gt_wh.to(device),
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


class MultiTaskLoss(nn.Module):
    """
        MTL多任务学习,自动权重调节: https://zhuanlan.zhihu.com/p/367881339
    """
    def __init__(self, cfg):
        super().__init__()
        self.criterion = DistanceLoss(loss_type='L2', reduction='mean')
        # self.smoothl1_loss = JointsDistanceLoss(use_target_weight=True,
        #                                         loss_type='smoothl1')
        self.loss_weight = cfg.LOSS.loss_weight
        self.auto_weight = cfg.LOSS.auto_weight
        if self.auto_weight:
            params = torch.ones(len(self.loss_weight), requires_grad=True)
            # TODO:将这个参数也放入优化器的参数优化列表中
            self.p = nn.Parameter(params, requires_grad=True)

    def forward(self, output, target, target_weight):
        device = output.device
        # print(f"{output.shape=}")
        # print(f"{target.shape=}")
        # print(f"{target_weight.shape=}")
        kpt_loss = self.criterion(output, target.to(device),
                                target_weight.to(device)) * self.loss_weight[0]

        loss_dict = dict(kpt_loss=kpt_loss.item())
        return kpt_loss, loss_dict

    # def forward(self, output, target, target_weight):
    #     kpt_loss, wh_loss = 0, 0
    #     device = output[-1].device
    #     pred_point, pred_wh = torch.split(output, (22, 2), dim=1)
    #     gt_point, gt_wh = torch.split(target, (22, 2), dim=1)
    #     weight_point, weight_wh = torch.split(target_weight, (22, 2), dim=1)

    #     kpt_loss += self.mse_loss(pred_point, gt_point.to(device),
    #                               weight_point.to(device)) * self.loss_weight[0]
    #     wh_loss += self.smoothl1_loss(pred_wh, gt_wh.to(device),
    #                                   weight_wh.to(device)) * self.loss_weight[1]

    #     loss = kpt_loss + wh_loss
    #     loss_dict = dict(kpt_loss=kpt_loss.item(), wh_loss=wh_loss.item())

    #     loss = 0    
    #     if self.auto_weight:
    #         for idx, loss in enumerate([kpt_loss, wh_loss]):
    #             c2 = self.p[idx] ** 2  # 正则项平方非负，后面再用log(1+c2)，使c2接近0
    #             loss = loss + 0.5 / c2 * loss + torch.log(1 + c2)
    #     else:
    #         loss = loss = kpt_loss + wh_loss
            
    #     return loss, loss_dict



