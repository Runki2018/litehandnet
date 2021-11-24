from re import S
import torch
from torch import nn
from loss.heatmapLoss import FocalLoss, RegionLoss, MaskLoss, L2Loss, JointsMSELoss
from config.config import config_dict as cfg

loss_func = {
    "Focal": FocalLoss(cfg["loss_alpha"], cfg["loss_beta"]),
    "MaskLoss": MaskLoss(),
    "RegionLoss": RegionLoss(),
    "MSE": nn.MSELoss(),
    "SmoothL1": nn.SmoothL1Loss(),
    'L2Loss': L2Loss(),
    'JointsLoss': JointsMSELoss(),
}

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
        self.kpt_loss = loss_func[cfg["kpt_loss"]]
        # self.mask_loss = loss_func[cfg["mask_loss"]]
        # self.region_loss = loss_func[cfg["region_loss"]]
        self.params = cfg["param"]
        self.n_joints = cfg['n_joints']
        self.n_out = cfg["nstack"] + 1

    def forward(self, hm_list, hm_gts, hm_weight=None):
        # loss_list = []

        # for i in range(self.n_out):
        #     if self.params[i] > 0:
        #         loss_list.append(
        #             self.params[i] * self.kpt_loss(hm_list[i], hm_gts, hm_weight))

        # # todo 到底是分开反向传播还是加起来反向传播好呢？ 可以做个实验
        # total_loss = sum(loss_list)

        # print(f"{total_loss.item()=}")
        # assert len(loss_list) == len(self.params), f"{len(loss_list)=} != {len(self.params)}"
        # loss_list = [l.item() for l in loss_list]

        total_loss = 0
        for i in range(self.n_out):
            total_loss = total_loss + \
                  self.params[i] * self.kpt_loss(hm_list[i], hm_gts, hm_weight)

        return total_loss


if __name__ == '__main__':
    from models.RKNet import HandNetSoftmax
    from data import Loader
    from config.config import DATASET

    model = HandNetSoftmax()
    model.load_state_dict(torch.load("../weight/0.693_mPCK_handnet1.pt")["model_state"])
    criterion = ()
    print("preparing data...")
    dataset, test_loader = Loader(batch_size=1, num_workers=1).test(
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
