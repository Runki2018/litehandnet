from turtle import forward
from grpc import Channel
import torch
from torch import group_norm, nn
from torch.nn import functional as F
from models import kaiming_init, constant_init, normal_init
from .common import channel_shuffle, ChannelAttension, SEBlock
from .repblocks import RepConv
import math
import numpy as np


class MSRB(nn.Module):
    def __init__(self, in_channels, out_channels, ca_type='none'):
        super().__init__()
        self.half_channels = in_channels // 2
        self.branch1 = nn.ModuleList([
            RepConv(self.half_channels, self.half_channels, 3, 1, 1,
                    groups=self.half_channels, activation=None),
            RepConv(self.half_channels, self.half_channels, 3, 1, 1,
                    groups=self.half_channels, activation=None)
             ])
        self.branch2 = nn.ModuleList([
            RepConv(self.half_channels, self.half_channels, 3, 1, 2, 2, 
                    groups=self.half_channels, activation=None),
            RepConv(self.half_channels, self.half_channels, 3, 1, 2, 2,
                    groups=self.half_channels, activation=None)
            ])

        # 信息交换，通道注意力模块
        if ca_type == 'se':
            self.ca = nn.ModuleList([
                SEBlock(out_channels, internal_neurons=out_channels // 16),
                SEBlock(out_channels, internal_neurons=out_channels // 16)])
        elif ca_type == 'ca':
            self.ca = nn.ModuleList([ChannelAttension(out_channels),
                                     ChannelAttension(out_channels)])
        else:
            self.ca = nn.ModuleList([nn.Identity(), nn.Identity()])
        
        self.conv = RepConv(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        out = x
        for _b1, _b2, _ca in zip(self.branch1, self.branch2, self.ca):
            left, right = torch.chunk(out, 2, dim=1)
            left = _b1(left)
            right = _b2(right)
            out = out + _ca(torch.cat([left, right], dim=1))
        return self.conv(out + x)

class RepBasicUnit(nn.Module):
    def __init__(self, in_channels, out_channels, ca_type='ca'):
        super(RepBasicUnit, self).__init__()
        self.left_part = in_channels // 2
        self.right_part_in = in_channels - self.left_part
        self.right_part_out = out_channels - self.left_part

        self.conv = nn.Sequential(
            RepConv(self.right_part_in, self.right_part_out, kernel=1),
            RepConv(self.right_part_out, self.right_part_out, kernel=3,
                    padding=1, groups=self.right_part_out),
        )
        if ca_type == 'se':
            self.ca = SEBlock(out_channels, internal_neurons=out_channels // 16)
        elif ca_type == 'ca':
            self.ca = ChannelAttension(out_channels)
        elif ca_type == 'none':
            self.ca = nn.Identity()
        else:
            raise ValueError(f'<{ca_type=}> not in se|ca|none')

    def forward(self, x):
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        out = self.conv(right)
        out = self.ca(torch.cat((left, out), 1))
        return out


class DWConv_ELAN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        mid_channel = in_channel // 2
        self.conv1 = nn.Sequential(
            RepConv(mid_channel, mid_channel, 3, 1, 1, groups=mid_channel, activation=None),
            RepConv(mid_channel, mid_channel, 1, 1, 0),
            RepConv(mid_channel, mid_channel, 3, 1, 1, groups=mid_channel, activation=None),
            RepConv(mid_channel, mid_channel, 1, 1, 0),
        )
        self.conv2 = nn.Sequential(
            RepConv(mid_channel, mid_channel, 3, 1, 1, groups=mid_channel, activation=None),
            RepConv(mid_channel, mid_channel, 1, 1, 0),
            RepConv(mid_channel, mid_channel, 3, 1, 1, groups=mid_channel, activation=None),
            RepConv(mid_channel, mid_channel, 1, 1, 0),
        )
        self.conv3 = nn.Conv2d(4 * mid_channel, out_channel, 1, 1, 0)
        self.c = mid_channel

    def forward(self, x):
        out1 = self.conv1(x[:, :self.c, :, :])
        out2 = self.conv2(out1)
        out = self.conv3(torch.cat([x, out1, out2], dim=1))
        out = channel_shuffle(out, groups=2)
        return out


class EncoderDecoder(nn.Module):
    def __init__(self, num_stage=4, channel=128, 
                 msrb_ca='ca', rbu_ca='ca'):
        super().__init__()
        self.num_stage = num_stage
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])

        self.maxpool = nn.MaxPool2d(2, 2)
        for i in range(num_stage):
            if i == 0:
                self.encoder.append(nn.Sequential(
                        MSRB(channel, channel,  ca_type=msrb_ca),
                        RepBasicUnit(channel, channel, ca_type=rbu_ca),
                    ))
                self.decoder.append(nn.Sequential(
                        MSRB(channel, channel, ca_type=msrb_ca),
                        RepBasicUnit(channel, channel, ca_type=rbu_ca),
                    ))
            else:
                self.encoder.append(nn.Sequential(
                        RepBasicUnit(channel, channel, ca_type=rbu_ca),
                        RepBasicUnit(channel, channel, ca_type=rbu_ca),
                        # DWConv_ELAN(channel, channel),
                        ))
                self.decoder.append(nn.Sequential(
                        RepBasicUnit(channel, channel, ca_type=rbu_ca),
                        RepBasicUnit(channel, channel, ca_type=rbu_ca),
                        # DWConv_ELAN(channel, channel),
                        ))

    def forward(self, x):
        out_encoder = []   # [128, 64, 32, 16, 8, 4]
        out_decoder = []   # [4, 8, 16, 32, 64, 128]

        # encoder 
        for i in range(self.num_stage):
            x = self.encoder[i](x)
            out_encoder.append(x)
            if i != self.num_stage - 1:
                x = self.maxpool(x)

        # decoder
        for i in range(self.num_stage-1, -1, -1):
            counterpart = out_encoder[i]
            if i == self.num_stage-1:
                x = self.decoder[i](counterpart)
                h, w = out_encoder[-1].shape[2:]
                shortcut = F.adaptive_avg_pool2d(out_encoder[0], (h, w))
                x = x + shortcut
            else:
                x = F.interpolate(x, size=counterpart.shape[2:])
                x = x + counterpart
                x = self.decoder[i](x)
            out_decoder.append(x)
        return tuple(out_decoder) 


class Stem(nn.Module):
    def __init__(self, channel):
        super().__init__()
        mid_channel = max(channel // 4, 32)
        self.conv1 = nn.Sequential(
            RepConv(3, mid_channel, 3, 2, 1),
            RepConv(mid_channel, mid_channel, 3, 1, 1, groups=mid_channel)
        )
        self.branch1 = nn.Sequential(
            RepConv(mid_channel, mid_channel, 1, 1, 0),
            RepConv(mid_channel, mid_channel, 3, 2, 1,
                    groups=mid_channel, activation=None),
            RepConv(mid_channel, mid_channel, 1, 1, 0),
        )
        self.branch2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        
        self.conv2 = nn.Sequential(
            RepConv(2*mid_channel, channel),
            RepBasicUnit(channel, channel),
            RepBasicUnit(channel, channel),
        )

    def forward(self, x):
        out = self.conv1(x)
        b1 = self.branch1(out)
        b2 = self.branch2(out)
        out = self.conv2(torch.cat([b1, b2], dim=1))
        return out


class LiteHandNet(nn.Module):
    def __init__(self, cfg, deploy=False):
        super().__init__()
        num_stage=cfg.MODEL.get('num_stage', 4)
        msrb_ca=cfg.MODEL.get('msrb_ca', 'ca')
        rbu_ca=cfg.MODEL.get('rbu_ca', 'ca')
        input_channel=cfg.MODEL.get('input_channel', 256)
        output_channel=cfg.MODEL.get('output_channel', cfg.DATASET.num_joints)

        self.deploy=deploy
        self.stem = Stem(input_channel)
        # self.stem = PatchifyStem(input_channel)
        self.backone = EncoderDecoder(num_stage, input_channel,
                                      msrb_ca=msrb_ca, rbu_ca=rbu_ca)

        self.neck = nn.Sequential(  
                RepBasicUnit(input_channel, input_channel),
                RepBasicUnit(input_channel, input_channel),
            )
        self.head = nn.Conv2d(input_channel, output_channel, 1, 1, 0)
        self.init_weights()

    def forward(self, x):
        out = self.stem(x)
        out_list = self.backone(out)
        out = self.neck(out_list[-1])
        out = self.head(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming_init(m)
                normal_init(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)

    def deploy_model(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        self.deploy = True
