import torch
import time
from torch import nn
from models.layers import Conv, Residual, Hourglass, BasicBlock, ConvBNReLu, SPP_layer
from models.hourglass import HourglassNet, Residual, Residual_SA
from models.attention import StageChannelAttention_fc, StageChannelAttention_all
from models.region_layer import SizeBlock, RegionBlock
from config.config import config_dict as cfg

torch.backends.cudnn.benchmark = True


class RANet(nn.Module):
    """
        加入通道注意力模块，算选每个阶段中最好的输出通道。
    """

    def __init__(self):
        super(RANet, self).__init__()
        n_points = cfg["n_joints"] # default 16
        main_channels = cfg["main_channels"]  # default 256
        num_hourglass = cfg["nstack"]  # default 2
        self.stage_network = HourglassNet(nstack=num_hourglass,
                                          inp_dim=main_channels,
                                          oup_dim=n_points,
                                          increase=0,
                                          basic_block=Residual)

        self.attention = StageChannelAttention_all(channel=n_points,
                                                   reduction=4,
                                                   n_block=num_hourglass,
                                                   min_unit=12)
        
        # self.attention = StageChannelAttention_fc(channel=n_points,
        #                                            n_block=num_hourglass)
        # self.init_weight()

    def forward(self, x):
        # print(x.size())
        hm_output = self.stage_network(x)  # [kpt1, kpt2, ..., kpt_nstack]

        kpt = self.attention(hm_output)
        hm_output.append(kpt)
        return hm_output

    # @staticmethod
    # def check(name, x):
    #     with torch.no_grad():
    #         print(f"{name=}, {x.shape=}")
    #         print(f"{x.isnan().sum()=}")
    #         print(f"{x.isinf().sum()=}")
    #         img_grid = vutils.make_grid(x.transpose(0, 1), normalize=True, scale_each=True,
    #                                     nrow=4)  # normalize进行归一化处理
    #         writer.add_image(name, img_grid, global_step=3)

    # def init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight.data, mean=0., std=0.05)
    #             # nn.init.normal_(m.bias.data, mean=0., std=0.05)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             # n = m.weight.shape[1]
    #             m.weight.data.normal_(0, 0.01)
    #             # m.bias.data.zero_()


if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    from torchvision import utils as vutils

    import cv2

    # image = cv2.imread("../test/test_example/2_256x256.jpg")
    image = cv2.imread("../data/mpii/images/017337360.jpg")
    image = cv2.resize(image, (256, 256), cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image)
    image = image / 255
    mean = image.mean()
    std = image.std()
    image = (image - mean) / std
    image = image.unsqueeze(dim=0)
    image = image.permute(0, 3, 1, 2)
    print(f"{image.shape=}")

    net = RANet()
    out = net(image)
    for hm in out:
        print(f'{id(hm)=}')
        print(f"{hm.shape=}")
    # net.load_state_dict(torch.load("../record/handnet3_mask_ca2_6/2021-11-15/0.078_ap_360epoch.pt")["model_state"])

    # writer = SummaryWriter("./Result")
    # img_grid = vutils.make_grid(image, normalize=True, scale_each=True, nrow=2)
    # # 绘制原始图像
    # writer.add_image('raw img', img_grid, global_step=0)  # j 表示feature map数

    # hm_kpts = net(image)
    # for hm in hm_kpts:
    #     print(f"{hm.shape=}")

    # writer.add_graph(net, image)
