import torch
from torch import nn
import math
import torch.nn.functional as F
# from thop import profile


def get_SRhandNet(cfg):
    """
    :input: (batch, 3, h, w)
    :ouput: (tuple)(
        (btach, 24, 22, 22),
        (btach, 24, 22, 22),
        (btach, 24, 44, 44),
        (btach, 24, 88, 88)
    )
    """
    # SRHandNet PyTorch Model: https://www.yangangwang.com/papers/WANG-SRH-2019-07.html
    model_path = 'models/pose_estimation/SRHandNet/srhandnet.pts'
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))

    # init model's parameters to train on FreiHand from scratch
    if not cfg.CHECKPOINT.resume:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0., std=0.05)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    return model


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 21, 3, 2, 1, dilation=1)
        self.conv2 = nn.Conv2d(3, 21, 3, 2, 2, dilation=2)
        self.conv3 = nn.Conv2d(3, 21, 3, 2, 5, dilation=5)
        
    def forward(self, x):
        out = []
        out.append(self.conv1(x))       # [Batch, 21, W, H]
        out.append(self.conv2(x))       # [Batch, 21, W, H]
        out.append(self.conv3(x))       # [Batch, 21, W, H]
        out = torch.cat(out, dim=1)    # [Batch, 63, W, H]
        return F.relu(out)

class BasicBlock(nn.Module):
    """以Caffe模型可视化后的结构为主, Caffe模型与hand.pts显示的模型参数不太一致
        hand.pts中,basicblock的skip conection必然有一个1x1卷积,而Caffe模型则不是。
        但是模型的通道数参考hand.pts.
    """
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
        )

        if stride == 2 or in_dim != out_dim:
            self.conv1x1 = nn.Conv2d(in_dim, out_dim, 1, stride, 0)
        else:
            self.conv1x1 = nn.Identity()

    def forward(self, x):
        out1 = self.conv3x3(x)
        out2 = self.conv1x1(x)
        return F.relu(out1 + out2)


class SRHandNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        out_c = cfg.MODEL.get('output_channel', 24)
        self.stem = Stem()
        
        self.block1 = nn.Sequential(
            BasicBlock(63, 128, stride=2),
            BasicBlock(128, 128)
        ) 
        self.block2 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256)
        )
        self.block3 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512)
        )
        self.block4 = nn.Sequential(
            BasicBlock(512, 256),
            BasicBlock(256, 128),
            nn.Conv2d(128, out_c, 1, 1, 0),         # OUT 1
            nn.ReLU()
        )
        self.block5 =nn.Sequential(
            BasicBlock(512+out_c, 256),
            BasicBlock(256, 128),
            nn.Conv2d(128, out_c, 1, 1, 0),         # OUT 2  => upsample 
            nn.ReLU()
        )
        self.block6 =nn.Sequential(
            BasicBlock(256+out_c, 256),
            BasicBlock(256, 128),
            nn.Conv2d(128, out_c, 1, 1, 0),         # OUT 3  => upsample
            nn.ReLU()
        )
        self.block7 =nn.Sequential(
            BasicBlock(128+out_c, 128),
            BasicBlock(128, 128),
            nn.Conv2d(128, out_c, 1, 1, 0),         # OUT 4
            nn.ReLU()
        )
        self.init_weights()

    def forward(self, x):
        x = self.stem(x)
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        b5 = self.block5(torch.cat([b3, b4], dim=1))
        b5_up = F.interpolate(b5, scale_factor=2)
        b6 = self.block6(torch.cat([b2, b5_up], dim=1))
        b6_up = F.interpolate(b6, scale_factor=2)
        b7 = self.block7(torch.cat([b1, b6_up], dim=1))
        return (b4, b5, b6, b7)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = get_SRhandNet()
    a = torch.rand(1, 3, 256, 256)
    out = net(a)
    print(f"{len(out)=}")
    for y in out:
        print(y.shape)
    # Pytorch 自带的计算参数量方法
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    


