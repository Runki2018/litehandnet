from math import e
from re import S
import torch
from torch import nn
from loss.heatmapLoss import FocalLoss, RegionLoss, MaskLoss, L2Loss, JointsMSELoss, SmoothL1Loss
from loss.centernet_simdr_loss import focal_loss, reg_l1_loss, KLDiscretLoss
import torch.nn.functional as F
from config.config import config_dict as cfg

loss_func = {
    "Focal": FocalLoss(cfg["loss_alpha"], cfg["loss_beta"]),
    "MaskLoss": MaskLoss(),
    "RegionLoss": RegionLoss(),
    "MSE": nn.MSELoss(),
    'L2Loss': L2Loss(),
    'SmoothL1Loss': SmoothL1Loss(),
    'JointsLoss': JointsMSELoss(),
}

class HMSimDRLoss(nn.Module):
    """
    MTL多任务学习，自动权重调节: https://zhuanlan.zhihu.com/p/367881339
    
    """
    def __init__(self, num_loss=4, auto_weights=False):
        super().__init__()
        self.hm_loss = loss_func[cfg["kpt_loss"]]
        self.vector_loss = KLDiscretLoss()
        params = torch.ones(num_loss, requires_grad=True)
        if auto_weights:
            self.p = nn.Parameter(params)   # TODO:将这个参数也放入优化器的参数优化列表中
        self.auto_weights = auto_weights
    
    def forward(self, hm, hm_gt,
                output_x, output_y,
                target_x, target_y, target_weight):
        
        hm_loss = 0
        for hm_i in hm:
            hm_loss += self.hm_loss(hm_i, hm_gt)
        vector_loss = self.vector_loss(output_x, output_y, target_x, target_y, target_weight)
        
        loss_sum = 0
        loss_dict = dict(hm=hm_loss, vector=vector_loss)
        if self.auto_weights:
            for i, loss in enumerate(loss_dict.values()):
                c2 = self.p[i] ** 2  # 正则项平方非负，后面再用log(1+c2)，使c2接近0
                loss_sum += 0.5 / c2 * loss + torch.log(1 + c2)
        else:
            loss_sum = sum([l for l in loss_dict.values()])
        
        loss_dict = {k:v.item() for k, v in loss_dict.items()}
        return loss_sum, loss_dict

class SimDRLoss(nn.Module):
    """
    MTL多任务学习，自动权重调节: https://zhuanlan.zhihu.com/p/367881339
    
    """
    def __init__(self):
        super().__init__()
        self.vector_loss = KLDiscretLoss()

    def forward(self,
                output_x, output_y,
                target_x, target_y, target_weight):
        vector_loss = self.vector_loss(output_x, output_y, target_x, target_y, target_weight)
        loss_dict = dict(vector=vector_loss)
        loss_dict = {k:v.item() for k, v in loss_dict.items()}
        return vector_loss, loss_dict


class HMLoss(nn.Module):
    """
        HandNet3 的损失函数，需要综合考虑中间监督和最终监督之间各种损失函数的选取，以及它们应该分别配置一个合理的权重占比
        目前改进的想法：region map中的中心点热图，与手部21个关键点一起预测。
        然后 22个点的热图和mask热图在深度方向上拼接在一起，用于预测region map 中的宽和高热图。
        mask背景热图上对于像手腕点那样的，远离其他点的离散点预测效果较差，而且密集区域值越大，这个对后面做乘积不太好，应该都是1才对。
        通道顺序： 0~1，宽高热图，2：背景热图， 3~24：中心点热图和21个关键点热图
    """

    def __init__(self):
        """
            初始函数里有三个超参数需要后期调整, a和b的值是参考cornerNet的， param的默认值是我根据监督的重要性和热图数目的差异设定的
         alpha: default 2， 这个值越大，正样本损失值越低，可以防止梯度爆炸
         beta: default 4，，这个值越大，负样本损失值月底，可以防止梯度爆炸
         param:  各个损失的占比权重
        """
        super(HMLoss, self).__init__()
        self.hm_loss = loss_func[cfg["kpt_loss"]]
        self.vector_loss = KLDiscretLoss()
        # self.mask_loss = loss_func[cfg["mask_loss"]]
        # self.region_loss = loss_func[cfg["region_loss"]]
        # self.params = cfg["param"]
        # self.n_out = len(self.params)

    def forward(self, pred_hm, gt_hm, 
                output_x=None, output_y=None,
                target_x=None, target_y=None,
                target_weight=None, cd=False):
        
        
        if not cd:  # 非循环训练，输入分辨率减半
            hm_loss = 0
            for hm in pred_hm:
                # region_loss += self.hm_loss(hm[:,:3], gt_hm[:, :3])
                # kpts_loss += self.hm_loss(hm[:, 3:], gt_hm[:, 3:], target_weight)
                hm_loss += self.hm_loss(hm, gt_hm)  
            loss = hm_loss
            loss_dict = dict(hm=hm_loss.item())
        else:
            vector_loss = self.vector_loss(output_x, output_y, target_x, target_y, target_weight)
            loss = vector_loss
            loss_dict = dict(vector=vector_loss.item())
            
        # loss = region_loss + kpts_loss + vector_loss
        # loss_dict = dict(region=region_loss.item(), kpts=kpts_loss.item(), vector=vector_loss.item())

        return loss, loss_dict

class HM_Region_Loss(nn.Module):
    """
        heatmap + region map 的损失韩式
        通道顺序： 0~2，region map (c, w, h)，3~23：21个关键点热图
    """

    def __init__(self):
        super(HM_Region_Loss, self).__init__()
        self.kpt_loss = loss_func[cfg["kpt_loss"]]
        # self.mask_loss = loss_func[cfg["mask_loss"]]
        # self.region_loss = loss_func[cfg["region_loss"]]

    def forward(self, hm_list, hm_gt, hm_weight=None):
        total_loss = 0

        for hm in hm_list:
            total_loss = total_loss + self.kpt_loss(hm, hm_gt, hm_weight)
        loss_dict = dict(hm=total_loss.item())

        return total_loss, loss_dict

if __name__ == '__main__':
    from models.RKNet import HandNetSoftmax
    from data import RegionLoader
    from config.config import DATASET

    model = HandNetSoftmax()
    model.load_state_dict(torch.load("../weight/0.693_mPCK_handnet1.pt")["model_state"])
    criterion = ()
    print("preparing data...")
    dataset, test_loader = RegionLoader(batch_size=1, num_workers=1).test(
        DATASET["root"], DATASET["train_file"])
    print("done!")

    for i, (img, target, label, bbox) in enumerate(test_loader):
        if i > 3:
            break

        k, m, r = model(img)
        tl, ll = criterion(k, m, r, target)

        print(f"{tl.item()=}")
        # ll = [p[k] * l.item() / tl.item() for k, l in enumerate(ll)]  # 观察各个监督的损失函数占比，用来决定各个损失前面的参数
        print(f"{ll=}")
