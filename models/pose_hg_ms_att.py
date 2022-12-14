import torch
from torch import nn
from torch.nn import functional as F
from models import kaiming_init, constant_init, normal_init


class DWConv(nn.Module):
    """DepthwiseSeparableConvModul 深度可分离卷积"""
    def __init__(self, in_channel, out_channel, stride=1, padding=1, dilation=1,mid_relu=True, last_relu=True, bias=False):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, stride, padding, groups=in_channel, bias=bias, dilation=dilation),
            nn.BatchNorm2d(in_channel))      
        self.mid_relu = nn.ReLU() if mid_relu else nn.Identity()  # 正常的DWConv直接有ReLU
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_channel))
        self.last_relu = nn.ReLU() if last_relu else nn.Identity()  # 正常的DWConv直接有ReLU
        
    def forward(self, x):
        out = self.mid_relu(self.depthwise_conv(x)) 
        out = self.last_relu(self.pointwise_conv(out)) 
        return out
 
class BottleNeck(nn.Module):
    """用于提高深度,但尽可能少地增加运算量, 不改变通道数"""
    def __init__(self, channel):
        super(BottleNeck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 1, 1, 0),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel // 4, 3, 1, 1),
            nn.BatchNorm2d(channel // 4),  
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel, 1, 1, 0), 
            nn.BatchNorm2d(channel),  
        )
    def forward(self, x):
        return F.relu(x + self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, stride=1):
        super(BasicBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, 3, stride, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),  
        )

        if stride == 2 or inp_dim != out_dim:
            self.skip_layer = nn.Sequential(
                nn.Conv2d(inp_dim, out_dim, 1, stride, 0),
                nn.BatchNorm2d(out_dim)
            )
        else:
            self.skip_layer = nn.Identity()
    def forward(self, x):
        return F.relu(self.skip_layer(x) + self.conv(x))

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim, stride=1, num_block=2):
        super().__init__()
        self.conv1 = BasicBlock(inp_dim, out_dim, stride)
        self.blocks = nn.Sequential(*[BottleNeck(out_dim) for _ in range(num_block)])

    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks(out)
        return out

class BRC(nn.Module):
    """  BN + Relu + Conv2d """    
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, dilation=1):
        super(BRC, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                            padding=padding, bias=bias, dilation=dilation)
        self.silu = nn.SiLU(inplace=True)
        self.bn = nn.BatchNorm2d(inp_dim)

    def forward(self, x):
        x = self.bn(x)
        x = self.silu(x)
        x = self.conv(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, num_levels=5, inp_dim=128, num_blocks=[]):
        super().__init__()
        self.num_levels = num_levels
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        assert len(num_blocks) == num_levels - 1

        self.encoder.append(ME_att(inp_dim, inp_dim))
        for i in range(num_levels-1):
            self.encoder.append(Residual(inp_dim, inp_dim, 2, num_blocks[i]))
            self.decoder.append(Residual(inp_dim, inp_dim))
        self.decoder.append(ME_att(inp_dim, inp_dim))

    def forward(self, x):
        out_encoder = []   # [128, 64, 32, 16, 8, 4]
        out_decoder = []   # [4, 8, 16, 32, 64, 128]

        # encoder 
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            out_encoder.append(x)

        # ! 我觉得只添加一次简单的shortcut就够了
        h, w = out_encoder[-1].shape[2:]
        shortcut = F.adaptive_avg_pool2d(out_encoder[0], (h, w))

        # decoder  
        for i, decoder_layer in enumerate(self.decoder):
            counterpart = out_encoder[self.num_levels-1-i]
            if i == 0:
                x = decoder_layer(counterpart)
                x = x + shortcut
            else:
                h, w = counterpart.shape[2:]
                x = decoder_layer(x)
                x = F.interpolate(x, size=(h, w))
                x = x + counterpart
            out_decoder.append(x)
        return tuple(out_decoder) 


class ME_att(nn.Module):
    """
    https://blog.csdn.net/KevinZ5111/article/details/104730835?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-4-104730835.pc_agg_new_rank&utm_term=block%E6%94%B9%E8%BF%9B+residual&spm=1000.2123.3001.4430
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        mid_c = in_c // 2
        self.conv1 = BRC(in_c, mid_c, 1, 1, 0)

        self.mid1_conv = nn.ModuleList([
            nn.Sequential(
                DWConv(mid_c, mid_c // 2),
                DWConv(mid_c // 2, mid_c // 2)
            ), 
            nn.Sequential(
                DWConv(mid_c, mid_c),
                DWConv(mid_c, mid_c),        
            )])

        self.mid2_conv = nn.ModuleList([
            nn.Sequential(
                DWConv(mid_c, mid_c // 2, dilation=2, padding=2),
                DWConv(mid_c // 2, mid_c // 2),),
            nn.Sequential(
                DWConv(mid_c, mid_c, dilation=2, padding=2),
                DWConv(mid_c, mid_c))
            ])

        self.conv2 = BRC(in_c, out_c, 1, 1, 0, bias=False)
        self.att = nn.Sequential(
                            nn.AdaptiveAvgPool2d((3,3)),
                            nn.BatchNorm2d(out_c),
                            nn.ReLU(),
                            nn.Conv2d(out_c, out_c, 3, 1, 0, groups=out_c),
                            nn.Flatten(),
                            nn.Dropout(p=0.3),
                            nn.Linear(out_c, out_c),
                            nn.Sigmoid(),  
                            )

    def forward(self, x):
        m = self.conv1(x)
        for i in range(2):
            m1 = self.mid1_conv[i](m)
            m2 = self.mid2_conv[i](m)
            m = torch.cat([m1, m2], dim=1)

        features = m + x
        out = self.conv2(features)
        b, c, _, _ = out.shape
        out = out * self.att(out).view(b, c, 1, 1)
        return out


class my_pelee_stem(nn.Module):
    """ 我在Conv1中再加了一个3x3卷积, 来提高stem的初始感受野"""
    def __init__(self, out_channel=256, min_mid_c=32):
        super().__init__()
        mid_channel = out_channel // 4 if out_channel // 4 >= min_mid_c else min_mid_c

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, mid_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, 1, 1,
                      groups=mid_channel, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, 1, 1, 0),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True), 
            nn.Conv2d(mid_channel, mid_channel, 3, 2, 1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True)
        )
        self.branch2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv1x1 = nn.Conv2d(mid_channel * 2, out_channel, 1, 1, 0)

    def forward(self, x):
        out = self.conv1(x)
        b1 = self.branch1(out)
        b2 = self.branch2(out)
        out = torch.cat([b1, b2], dim=1)
        out = self.conv1x1(out)
        return out


class MultiScaleAttentionHourglass(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_stage=cfg.MODEL.get('num_stage', 4)
        inp_dim=cfg.MODEL.get('input_channel', 128)
        oup_dim=cfg.MODEL.get('output_channel', cfg.DATASET.num_joints)
        num_block=cfg.MODEL.get('num_block', [2, 2, 2])
        self.with_activation = cfg.MODEL.get('output_acitivation', False)

        self.pre = my_pelee_stem(inp_dim)
        self.hgs = EncoderDecoder(num_stage, inp_dim, num_block) 

        self.features = nn.Sequential(
                BottleNeck(inp_dim),
                nn.Conv2d(inp_dim, inp_dim, 1, 1, 0),
                nn.BatchNorm2d(inp_dim),
                nn.LeakyReLU(),
            )

        # self.out_layer = nn.Conv2d(inp_dim, oup_dim, 1, 1, 0)
        self.outs = nn.Conv2d(inp_dim, oup_dim, 1, 1, 0)
        self.init_weights()

    def forward(self, imgs):
        # our posenet
        x = self.pre(imgs)
        hg = self.hgs(x)
        feature = self.features(hg[-1])
        preds = self.outs(feature)

        if self.with_activation:  # 加快模型收敛
            preds = F.leaky_relu(preds, 0.5)
        return preds
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming_init(m)
                normal_init(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)

    
