import torch
from torch import nn
from torch.nn import functional as F
from models import kaiming_init, constant_init, normal_init
from .common import SEBlock, channel_shuffle, ChannelAttension
from .repblocks import RepConv, RepBlock

class DWConv(nn.Module):
    """DepthwiseSeparableConvModul 深度可分离卷积"""
    def __init__(self, in_channel, out_channel, stride=1, padding=1, dilation=1,
                 activation=nn.LeakyReLU):
        super().__init__()
        self.depthwise_conv = RepConv(in_channel, in_channel, 3, stride, padding,
                                       groups=in_channel, dilation=dilation,
                                       activation=activation, inplace=False)
        self.pointwise_conv = RepConv(in_channel, out_channel, 1, 1, 0,
                                      activation=activation, inplace=False)
    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out

class BottleNeck(nn.Module):
    """用于提高深度,但尽可能少地增加运算量, 不改变通道数"""
    def __init__(self, channel, reduction=4, activation=nn.LeakyReLU):
        super(BottleNeck, self).__init__()
        mid_channel = channel // reduction
        self.conv = nn.Sequential(
            RepConv(channel, mid_channel, 1, 1, 0,
                    activation=activation, inplace=True),
            RepConv(mid_channel, mid_channel, 3, 1, 1,
                     activation=activation, inplace=True),
            RepConv(mid_channel, channel, 1, 1, 0, activation=None),
        )
        self.activation = activation()
    def forward(self, x):
        return self.activation (x + self.conv(x))

class BasicBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, stride=1, activation=nn.LeakyReLU):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            RepConv(inp_dim, out_dim, 3, stride, 1,
                     activation=activation, inplace=True),
            RepConv(inp_dim, out_dim, 3, 1, 1,
                     activation=None)
        )
        if stride == 2 or inp_dim != out_dim:
            self.skip_layer = RepConv(inp_dim, out_dim, 1, stride, 0, activation=None)
        else:
            self.skip_layer = nn.Identity()
        self.activation = activation()
    def forward(self, x):
        return self.activation(self.skip_layer(x) + self.conv(x))


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim, stride=2, num_block=2,
                 reduction=2, activation=nn.LeakyReLU):
        super().__init__()
        self.conv1 = BasicBlock(inp_dim, out_dim, stride, activation)
        self.blocks = nn.Sequential(
            *[BottleNeck(out_dim, reduction, activation) for _ in range(num_block)])

    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks(out)
        return out


class EncoderDecoder(nn.Module):
    def __init__(self, num_levels=5, inp_dim=128, num_blocks=[],
                 ca_type='ca', reduction=2, activation=nn.LeakyReLU):
        super().__init__()
        self.num_levels = num_levels
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        assert len(num_blocks) == num_levels - 1

        self.encoder.append(MSAB(inp_dim, inp_dim, ca_type=ca_type))
        for i in range(num_levels-1):
            self.encoder.append(
                Residual(inp_dim, inp_dim, 2, num_blocks[i], reduction, activation))
            self.decoder.append(
                Residual(inp_dim, inp_dim, 1, num_blocks[i], reduction, activation))
        self.decoder.append(MSAB(inp_dim, inp_dim, ca_type=ca_type))

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


class MSAB(nn.Module):
    """
    https://blog.csdn.net/KevinZ5111/article/details/104730835?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-4-104730835.pc_agg_new_rank&utm_term=block%E6%94%B9%E8%BF%9B+residual&spm=1000.2123.3001.4430
    """
    def __init__(self, in_c, out_c, ca_type='ca', activation=nn.LeakyReLU):
        super().__init__()

        mid_c = in_c // 2
        self.conv1 = RepConv(in_c, mid_c, 1, 1, 0, activation=activation, inplace=True)

        self.mid1_conv = nn.ModuleList([
            nn.Sequential(
                DWConv(mid_c, mid_c // 2, activation=activation),
                DWConv(mid_c // 2, mid_c // 2, activation=activation)
            ), 
            nn.Sequential(
                DWConv(mid_c, mid_c, activation=activation),
                DWConv(mid_c, mid_c, activation=activation),        
            )])

        self.mid2_conv = nn.ModuleList([
            nn.Sequential(
                DWConv(mid_c, mid_c // 2, dilation=2, padding=2, activation=activation),
                DWConv(mid_c // 2, mid_c // 2, activation=activation)),
            nn.Sequential(
                DWConv(mid_c, mid_c, dilation=2, padding=2, activation=activation),
                DWConv(mid_c, mid_c, activation=activation))
            ])

        self.conv2 = RepConv(in_c, out_c, 1, 1, 0, activation=activation, inplace=True)
        
        if ca_type == 'se':
            self.ca = SEBlock(out_c, internal_neurons=out_c // 16)
        elif ca_type == 'ca':
            self.ca = ChannelAttension(out_c)
        elif ca_type == 'none':
            self.ca = nn.Identity()
        else:
            raise ValueError(f'<{ca_type=}> not in se|ca|none')

    def forward(self, x):
        m = self.conv1(x)
        for i in range(2):
            m1 = self.mid1_conv[i](m)
            m2 = self.mid2_conv[i](m)
            m = torch.cat([m1, m2], dim=1)

        features = m + x
        out = self.conv2(features)
        out = self.ca(out)
        return out


class Stem(nn.Module):
    """ 我在Conv1中再加了一个3x3卷积, 来提高stem的初始感受野"""
    def __init__(self, out_channel=256, min_mid_c=32, activation=nn.LeakyReLU):
        super().__init__()
        mid_channel = out_channel // 4 if out_channel // 4 >= min_mid_c else min_mid_c

        self.conv1 = nn.Sequential(
            RepBlock(3, mid_channel, 3, 2, 1, activation=activation, inplace=True),
            RepBlock(mid_channel, mid_channel, 7, 1, 3, groups=mid_channel,
                     activation=activation, inplace=True)
        )
        self.branch1 = nn.Sequential(
            RepConv(mid_channel, mid_channel, 1, 1, 0, activation=activation, inplace=True),
            RepConv(mid_channel, mid_channel, 3, 2, 1, activation=activation, inplace=True),
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


class LiteHandNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_stage=cfg.MODEL.get('num_stage', 4)
        inp_dim=cfg.MODEL.get('input_channel', 128)
        oup_dim=cfg.MODEL.get('output_channel', cfg.DATASET.num_joints)
        num_block=cfg.MODEL.get('num_block', [2, 2, 2])
        ca_type = cfg.MODEL.get('ca_type', 'ca')  # 'ca' | 'se' | 'none'
        reduction = cfg.MODEL.get('reduction', 2)
        activation = cfg.MODEL.get('activation', 'LeakyReLU')
        assert reduction in [2, 4]
        assert ca_type in ['ca', 'se', 'none']
        assert activation.lower() in ['leakyrelu', 'relu', 'silu']
        if activation.lower() == 'leakyrelu':
            activation = nn.LeakyReLU
        elif activation.lower() == 'relu':
            activation = nn.ReLU
        else:
            activation = nn.SiLU

        self.pre = Stem(inp_dim, activation=activation)
        self.hgs = EncoderDecoder(num_stage, inp_dim, num_block,
                                  ca_type, reduction, activation) 

        self.features = nn.Sequential(
                BottleNeck(inp_dim, 2, activation),
                RepConv(inp_dim, inp_dim, 1, 1, 0, activation=activation, inplace=True),
            )

        self.out_layer = nn.Conv2d(inp_dim, oup_dim, 1, 1, 0)
        self.init_weights()

    def forward(self, imgs):
        # our posenet
        x = self.pre(imgs)
        hg = self.hgs(x)
        feature = self.features(hg[-1])
        preds = self.out_layer(feature)
        return preds

    def init_weights(self):
        for m in self.modules():
            normal_init(m)

    def deploy_model(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        self.deploy = True