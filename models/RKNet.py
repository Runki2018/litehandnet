import torch
from torch import nn
from models.layers import Conv, Residual, Hourglass, BasicBlock, ConvBNReLu
from config.config import config_dict as cfg


class RegionKeypointNetwork(nn.Module):
    """
        将关键点输出以sigmoid函数来计算概率的形式计算，在handnet1的基础上改进注意力模块，背景和region maps预测模型, 各层的通道数改变
        目前改进的想法：region map中的中心点热图，与手部21个关键点一起预测。
        然后 22个点的热图和mask热图在深度方向上拼接在一起，用于预测region map 中的宽和高热图。
        mask背景热图上对于像手腕点那样的，远离其他点的离散点预测效果较差，而且密集区域值越大，这个对后面做乘积不太好，应该都是1才对。
        通道顺序： 0~1，宽高热图，2：背景热图， 3~24：中心点热图和21个关键点热图
    """

    def __init__(self):
        super(RegionKeypointNetwork, self).__init__()
        self.hm_size = cfg["hm_size"]
        self.n_points = cfg["n_joints"] + 1  # default 22, 1个中心点 + 21个关键点
        main_channels = cfg["main_channels"]  # default 64
        self.num_hourglass = cfg["num_hourglass"]  # default 2

        self.stem = nn.Sequential(  # todo: 这里通道数 64 ， channel_mid的最佳值要实验确定
            ConvBNReLu(3, main_channels // 2, 3, 2, 1),
            ConvBNReLu(main_channels // 2, main_channels, 3, 2, 1),
        )  # stem网络层，提取初级特征， 输出是 (batch, main_channels, height/4, width/4)

        # 两个沙漏模块，初步预测关键点
        self.hourglass_modules = nn.ModuleList([
            nn.Sequential(
                Hourglass(n=3, f=main_channels, increment=0),
            ) for _ in range(self.num_hourglass)
        ])

        # 公用 融合关键件检测器和主特征通路
        self.merge = nn.ModuleList([
            Conv(self.n_points, main_channels, bn=False, relu=False) for _ in range(self.num_hourglass)
        ])

        # refine分支1， 用于预测手部区域和预测bbox
        self.region_stem = nn.Sequential(  # 预测手部区域和region maps 的stem特征层
            Residual(main_channels, main_channels // 2),
            BasicBlock(main_channels // 2, main_channels // 2),  # 只有通道数不变时，才能用 BasiceBlock, 否则要加下采样块
        )

        self.hand_segmentation = nn.Sequential(  # 预测手部的背景热图 或 mask
            Residual(main_channels // 2, main_channels // 4),  # 32, 16
            Residual(main_channels // 4, main_channels // 8),  # 16, 8
            Conv(main_channels // 8, 1, 1, bn=True, relu=False),
            nn.Sigmoid(),
        )

        self.get_region = nn.Sequential(  # 预测 region maps 的宽高热图
            Residual(main_channels // 2, main_channels // 4),  # 32, 16
            Conv(main_channels // 4, 2, bn=True, relu=False),
            nn.Sigmoid(),
        )

        # refine分支2， 用于优化关键点预测
        self.attention_factor = nn.Sequential(  # todo 注意力模块是关键，怎么添加一些有效的层
            Residual(main_channels // 2, main_channels // 8),  # 64, 32
            Conv(main_channels // 8, 1, bn=True, relu=True),
            # nn.Sigmoid(),
        )

        self.get_keypoints = nn.ModuleList([
            nn.Sequential(  # 共用一个获取关键点的解码器
                Residual(main_channels, main_channels // 2),
                Residual(main_channels // 2, self.n_points),
                nn.Sigmoid(),
            ) for _ in range(self.num_hourglass + 1)
        ])
        self.init_weight()

    def forward(self, x):
        # print(x.size())
        hm_output = []  # 关键点热图输出
        batch = x.shape[0]
        stem = self.stem(x)
        # self.check("stem", stem)

        x = stem + torch.ones_like(stem) * 0.5
        for i in range(self.num_hourglass):
            x = self.hourglass_modules[i](x)
            kpt = self.get_keypoints[i](x)
            hm_output.append(kpt)
            # self.check("kpt{}".format(i), kpt)
            merge_feature = self.merge[i](kpt)
            x = merge_feature + x + stem

        # refine分支1： 预测手部分割掩膜 和 region maps
        region_stem = self.region_stem(x)
        mask = self.hand_segmentation(region_stem)
        # self.check("mask", mask)

        # refine分支2： 优化关键点预测
        af = self.attention_factor(mask+region_stem)
        # self.check("af", af)
        out = x * af
        kpt = self.get_keypoints[-1](out)
        hm_output.append(kpt)
        # self.check("kpt_final", kpt)

        region_maps = self.get_region(region_stem)
        # self.check("region", region_maps)
        hm_output.append(mask)
        hm_output.append(region_maps)
        return hm_output

    @staticmethod
    def check(name, x):
        with torch.no_grad():
            print(f"{name=}, {x.shape=}")
            print(f"{x.isnan().sum()=}")
            print(f"{x.isinf().sum()=}")
            img_grid = vutils.make_grid(x.transpose(0, 1), normalize=True, scale_each=True,
                                        nrow=4)  # normalize进行归一化处理
            writer.add_image(name, img_grid, global_step=3)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0., std=0.05)
                # nn.init.normal_(m.bias.data, mean=0., std=0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[1]
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    from torchvision import utils as vutils
    import cv2

    image = cv2.imread("../image/00000000.jpg")
    image = cv2.resize(image, (352, 352), cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image)
    image = image / 255
    mean = image.mean()
    std = image.std()
    image = (image - mean) / std
    image = image.unsqueeze(dim=0)
    image = image.permute(0, 3, 1, 2)
    print(f"{image.shape=}")

    net = RegionKeypointNetwork()
    # net.load_state_dict(torch.load("../weight/0.693_mPCK_handnet1.pt")["model_state"])

    writer = SummaryWriter("./Result")
    # img_grid = vutils.make_grid(image, normalize=True, scale_each=True, nrow=2)
    # # 绘制原始图像
    # writer.add_image('raw img', img_grid, global_step=0)  # j 表示feature map数

    # state = net.state_dict()
    # keys = state.keys()
    # for key in keys:  # 基本上都是 -0.1~+0.1
    #     if not 'conv' in key:
    #         continue
    #     print("-" * 100)
    #     print(f"{key=}".center(20, '-'))
    #     a = state[key].type(torch.float32)
    #     print(f"{a.mean()=}")
    #     print(f"{a.std()=}")
    #     print(f"{a.max()=}")
    #     print(f"{a.min()=}")

    hm_kpts, m, r = net(image)
    print(f"{m.shape=}")
    print(f"{r.shape=}")
    for hm_k in hm_kpts:
        print(f"{hm_k.shape=}")
        print(f"{id(hm_k)=}")

    writer.add_graph(net, image)
