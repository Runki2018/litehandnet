import torch
import torch.nn.functional as F
import torch.nn as nn


class KLDiscretLoss(nn.Module):
    """计算预测x，y的 1d vector的损失。

    Args:
        nn ([type]): [description]
    """
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        # self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        # self.criterion_ = nn.KLDivLoss(reduction='none')
        # self.softmax = nn.Softmax(dim=1)
        self.criterion_ = nn.SmoothL1Loss(reduction='mean')

        
    def criterion(self, dec_outs, labels):
        # scores = self.LogSoftmax(dec_outs)
        # gt_scorces = self.softmax(labels)
        # loss = torch.mean(self.criterion_(scores, gt_scorces), dim=1)
        loss = self.criterion_(dec_outs, labels)
        return loss

    def forward(self, output_x, output_y, target_x, target_y, target_weight):
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx].squeeze()
            coord_y_pred = output_y[:, idx].squeeze()
            coord_x_gt = target_x[:, idx].squeeze()
            coord_y_gt = target_y[:, idx].squeeze()
            weight = target_weight[:, idx].squeeze()
            loss += (self.criterion(coord_x_pred, coord_x_gt).mul(weight).mean())
            loss += (self.criterion(coord_y_pred, coord_y_gt).mul(weight).mean())
        return loss / num_joints

# def ae_loss(tag_x, tag_y, mask):
#     # TODO: 如何实现这个pull和push函数呢？
#     num = mask.sum(dim=1, keepdim=True).float()
     



def focal_loss(pred, target):
    """计算 center 热图的损失

    Args:
        pred (tensor): (b, 1, h, w)
        target (tensor): (b, 1, h, w)

    Returns:
        tensor: scalar (1)
    """

    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    
    # -------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    # -------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    # -------------------------------------------------------------------------#
    #   进行损失的归一化
    # -------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def reg_l1_loss(pred, target, mask):
    """ 计算 宽高 和 偏置 的 L1Loss, 通道含义分别是 (w_bbox, h_bbox, w_offset, h_offset)

    Args:
        pred (tensor): (b, 4, h, w)
        target (tensor): (b, 4, h, w)
        mask (tensor): (b, 1, h, w) 记录中心点位置的mask, 存在中心点为1，其余为0

    Returns:
        tensor: scalar (1)
    """
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

