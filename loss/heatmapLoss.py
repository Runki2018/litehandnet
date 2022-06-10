import torch
from torch import nn


class KLFocalLoss(nn.Module):
    """
        参考cornerNet损失函数进行了修改
        https://github.com/feifeiwei/Pytorch-CornerNet/blob/master/module/loss_module.py
    """

    def __init__(self):
        super(KLFocalLoss, self).__init__()
        self.Softmax = nn.Softmax(dim=2)
        self.LogSoftmax = nn.LogSoftmax(dim=2)  # [B,LOGITS]
        self.KLDiv = nn.KLDivLoss(reduction='none')
        
        
    def criterion(self, hm_preds, hm_gts):
        b, c, _, _ = hm_preds.shape
        hm_preds = hm_preds.view(b, c, -1)
        hm_gts = hm_gts.view(b, c, -1)
        
        log_scores = self.LogSoftmax(hm_preds)
        gt_scorces = self.Softmax(hm_gts)
        
        loss = self.KLDiv(log_scores, gt_scorces)
        loss = torch.sum(loss, dim=2)
        return loss

    def forward(self, hm_preds, hm_gts, hm_weight=None):
        """

        :param hm_preds: (batch, n_joints or 1+n_joints, hm_height, hm_width)
        :param hm_gts: (batch, n_joints or 1+n_joints, hm_height, hm_width)
        :param hm_weight: (batch, n_joints or 1+n_joints, 1)
        :return:
        """
        if hm_weight is None:
            loss = self.criterion(hm_preds, hm_gts).mean()
        else:
            hm_weight = hm_weight.view(hm_weight.shape[0], -1)
            loss = self.criterion(hm_preds, hm_gts).mul(hm_weight).mean()

        return loss



class FocalLoss(nn.Module):
    """
        参考cornerNet损失函数进行了修改
        https://github.com/feifeiwei/Pytorch-CornerNet/blob/master/module/loss_module.py
    """

    def __init__(self, alpha=2, beta=0.25, thr=0.4):
        super(FocalLoss, self).__init__()
        self.a = alpha  # 2的倍数, 类似于欧式距离衡量，预测值和真实值的差距
        self.b = beta  # (0~1)
        self.r = 0.25  # (0~1)
        self.thr = thr  # todo 真值热图中置信度大于该阈值的点为正样本， 确定该阈值的大小

    def forward(self, hm_preds, hm_gts, hm_weight=None):
        """

        :param hm_preds: (batch, n_joints or 1+n_joints, hm_height, hm_width)
        :param hm_gts: (batch, n_joints or 1+n_joints, hm_height, hm_width)
        :param hm_weight: (batch, n_joints or 1+n_joints, 1)
        :return:
        """
        pos_mask = hm_gts.gt(self.thr)  # todo 我把非零部分都当作是真值区域，验证其有效性
        neg_mask = ~pos_mask

        loss = 0
        for batch_idx, (pred, gt) in enumerate(zip(hm_preds, hm_gts)):
            for joint_idx, (pred_, gt_) in enumerate(zip(pred, gt)):
                if hm_weight is not None and hm_weight[batch_idx, joint_idx] == 0:
                    continue

                pos_pred = pred_[pos_mask[batch_idx, joint_idx]]  # 正样本预测为真的概率，越高越好
                neg_pred = 1 - pred_[neg_mask[batch_idx, joint_idx]]  # 负样本预测预测为假概率，越高越好

                pos_pred = pos_pred.clip(1e-30, 1)  # 限制在0~1的函数，如sigmoid函数，
                neg_pred = neg_pred.clip(1e-30, 1)

                distance = torch.pow(gt_ - pred_, self.a)  # 预测值和真值的的绝对差的偶次方, a=2类似于欧斯距离
                # todo 要同时保证
                #  1) 在p=1时，loss=0； p=0时， loss最大，
                #  2) p在 (0,1]之间，保证 log函数有意义
                #  3)保证梯度足够，加快收敛速度，但不能出现loss=inf，导致输出为nan
                #  4)确定合适的的超参数 a和 b

                pos_loss = self.r * torch.log(pos_pred) * distance[pos_mask[batch_idx, joint_idx]]

                neg_loss = (1 - self.r) * torch.log(neg_pred) * distance[neg_mask[batch_idx, joint_idx]]

                num_pos = pos_mask[batch_idx, joint_idx].float().sum()
                pos_loss = pos_loss.sum()
                neg_loss = neg_loss.sum()

                if pos_pred.nelement() == 0:  # no element
                    loss = loss - neg_loss 
                else:
                    loss = loss - (pos_loss + neg_loss) / num_pos

                # if torch.isnan(loss):
                #     import pdb
                #     pdb.set_trace()

        return loss


class MaskLoss(nn.Module):
    """其实就是交叉熵损失函数, 用于计算mask3的损失, 输入必须是0~1"""

    def __init__(self, a=0.5, thr=0.2):
        super(MaskLoss, self).__init__()
        self.a = a  # todo 确定超参数 a
        self.thr = thr

    def forward(self, pred_hms, gt_hms):
        pos_mask = gt_hms.gt(self.thr)
        neg_mask = ~pos_mask

        pos_pred = (pred_hms[pos_mask] + 1 - gt_hms[pos_mask]).clip(1e-30, 1)  # 限制在0~1的函数，如sigmoid函数，
        neg_pred = (1 - pred_hms[neg_mask]).clip(1e-30, 1)

        pos_loss = torch.log(pos_pred)
        neg_loss = (1 - gt_hms[neg_mask]) * torch.log(neg_pred)

        num = pos_mask.sum()
        loss = -1.0 * (pos_loss.sum() + self.a * neg_loss.sum()) / num

        if torch.isnan(loss):
            import pdb
            pdb.set_trace()

        return loss


class RegionLoss(nn.Module):
    def __init__(self, a=0.5, thr=0.):
        super(RegionLoss, self).__init__()
        self.a = a  # todo 确定超参数 a
        self.const = 4 / (3.14159 ** 2)
        self.thr = thr

    def forward(self, pred_hms, gt_hms):
        pos_mask = gt_hms.gt(self.thr)
        if pos_mask.sum() == 0:
            return torch.zeros(1).to(pred_hms.device)
        neg_mask = ~pos_mask

        pos_pred = (pred_hms[pos_mask]).clip(1e-30, 1)  # 限制在0~1的函数，如sigmoid函数，
        neg_pred = (1 - pred_hms[neg_mask]).clip(1e-30, 1)
        pos_gt = gt_hms[pos_mask]  # 正样本真值

        # 注意保证，下面两项loss都是负数，且y=p时，loss最小。
        # pos_loss = (pos_gt - pos_pred) * torch.log(pos_pred / pos_gt)  # 因为宽高热图的真值0~1，通过分数来限制。
        # 因为宽高热图的真值0~1，通过分数来限制。 学习YOLOv3中平方根的方法来加大小框损失，减少大框损失
        pos_loss = (torch.sqrt(pos_gt) - torch.sqrt(pos_pred)) * torch.log(pos_pred / pos_gt)
        neg_loss = torch.log(neg_pred)

        pos_num = pos_mask.sum()
        loss = -1.0 * (pos_loss.sum() + self.a * neg_loss.sum()) / pos_num

        # w/h aspect-ratio loss
        pred_ratio = pred_hms[:, 0][pos_mask[:, 0]] / (pred_hms[:, 1][pos_mask[:, 1]] + 1e-6)
        gt_ratio = gt_hms[:, 0][pos_mask[:, 0]] / (gt_hms[:, 1][pos_mask[:, 1]] + 1e-6)
        aspect_loss = torch.arctan(pred_ratio) - torch.arctan(gt_ratio)
        aspect_loss = self.const * torch.pow(aspect_loss, 2)
        loss = loss + aspect_loss.mean()
        return loss


# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
class JointsDistanceLoss(nn.Module):
    def __init__(self, use_target_weight=True, loss_type='mse'):
        """
        MSE/MAE/SmoothL1 loss between output and GT body joints

        Args:
            use_target_weight (bool): use target weight.
                WARNING! This should be always true, otherwise the loss will sum the error for non-visible joints too.
                This has not the same meaning of joint_weights in the COCO dataset.
        """
        super(JointsDistanceLoss, self).__init__()
        assert loss_type.lower() in ['mse', 'mae', 'smoothl1']
        if loss_type.lower() == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        elif loss_type.lower() == 'mae':
            self.criterion = nn.L1Loss(reduction='mean')
        else:
            self.criterion = nn.SmoothL1Loss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        """

        Args:
            output (tensor): [N, K, H, W]
            target (tensor): [N, K, H, W]
            target_weight (tensor): [N, K, 1]

        Returns:
            loss: scalar
        """
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                if target_weight is None:
                    raise NameError
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class DistanceLoss(nn.Module):
    def __init__(self, loss_type='L2', reduction='mean'):
        super().__init__()
        if loss_type.lower() == 'l2':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type.lower() == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        else:
            self.criterion = nn.SmoothL1Loss(reduction='none')
        
        assert reduction in ['mean', 'sum', None], f"Error: {reduction=}"
        self.reduction = reduction

    def forward(self, output, target, target_weight):
        """
        Args:
            output (tensor): [N, K, H, W]
            target (tensor): [N, K, H, W]
            target_weight (tensor): [N, K, 1]
        """
        loss = self.criterion(output, target)
        loss *= target_weight.unsqueeze(-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


if __name__ == '__main__':
    pass
    img = torch.zeros((1, 3, 3))
    img[0, 1, 1] = 0.1
    img[0, 0, 0] = 0.5
    hms = torch.stack((img, img), dim=0)
    hm_gts = torch.zeros_like(hms)
    hm_gts[:, :, 1, 1] = 1

    # focal_loss = FocalLoss(alpha=2, beta=4)
    # l = focal_loss(hms, hm_gts)
    # sl = MaskLoss()
    sl = RegionLoss()
    l = sl(hms, hm_gts)

    print(hms)
    print(hms.shape)
    print(l)
