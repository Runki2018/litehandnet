import torch
from torch import nn
from torch.distributed.distributed_c10d import is_nccl_available
import torch.nn.functional as F
from torch.nn.modules import module, transformer
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from models.layers import DWConv, channel_shuffle



class SpatialWeighting(nn.Module):
    """类似于SENet的通道注意力加权 """
    def __init__(self, channels: int, ratio=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        mid_channels = int(channels / ratio)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, 1),
            nn.ReLU(True),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, channels, 1, 1),
            nn.ReLU(True),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out

class CrossResolutionWeighting(nn.Module):
    """用类似于SENet的方式，给不同分辨率的分支加权"""
    def __init__(self, channels: list, ratio=16):
        super().__init__()
        self.channels = channels  # list[40, 80, 160, 320]
        total_channel = sum(channels)

        mid_channel = int(total_channel / ratio)
        self.conv1 = nn.Sequential(
            nn.Conv2d(total_channel, mid_channel, 1, 1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, total_channel, 1, 1),
            nn.BatchNorm2d(total_channel),
            nn.ReLU(True),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x = [tensor(N, C1, H1, W1), tensor(N, C2, H2, W2), ...]
        # 将所有高分辨率的特征图通过aap池化到最低分辨率的特征图
        mini_size = x[-1].shape[-2:]  # [h, w]
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.shape[-2:], mode='nearest')
            for s, a in zip(x, out)
        ]
        return out

class ConditionalChannelWeighting(nn.Module):
    def __init__(self, in_channels, reduce_ratio, stride=1):
        super().__init__()
        branch_channels = [c // 2 for c in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            channels=branch_channels, ratio=reduce_ratio  
        )
        self.depthwise_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, stride, 1, groups=c),
                nn.BatchNorm2d(c)
            ) for c in branch_channels
        ])
        self.spatial_weighting = nn.ModuleList([
            SpatialWeighting(channels=c, ratio=4)
            for c in branch_channels
        ])
    
    def forward(self, x):
        x = [s.chunk(2, dim=1) for s in x]
        x1 = [s[0] for s in x]
        x2 = [s[1] for s in x]

        x2 = self.cross_resolution_weighting(x2)
        x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
        x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

        out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
        out = [channel_shuffle(s, 2) for s in out]
        return out

class StageModule(nn.Module):
    def __init__(self, in_branches, num_blocks,
     in_channels, reduce_ratio=8, with_fuse=True):
        super().__init__()
        self.in_branches = in_branches
        # self.out_branches = out_branches
        self.in_channels = in_channels
        self.with_fuse = with_fuse

        self.layers = nn.Sequential(*[
            ConditionalChannelWeighting(in_channels, reduce_ratio)
         for _ in range(num_blocks) ])
        
        if self.with_fuse and self.in_branches > 1:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()
        else:
            self.with_fuse = False
    
    def _make_fuse_layers(self):
        cs = self.in_channels
        fuse_layers = nn.ModuleList()  # 融合层模块列表
        for i in range(self.in_branches):  # 输出层
            fuse_layers.append(nn.ModuleList())
            for j in range(self.in_branches):  # 输入层
                c_in, c_out = cs[j], cs[i] 
                if i == j:  # 本层不变
                    fuse_layers[-1].append(nn.Identity())
                    # fuse_layers[-1].append(None)
                elif j > i:  # upsample    
                    fuse_layers[-1].append(nn.Sequential(
                            nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(c_out),
                            nn.Upsample( scale_factor=2**(j - i), mode='nearest')
                    ))
                else:  # downsample
                    conv_downsamples = []  # 一个下采样 i-j 次，第i-j次变通道
                    for k in range(i - j - 1):
                        conv_downsamples.append(DWConv(c_in, c_in, stride=2, mid_relu=False, last_relu=False))
                    # when k = i - j - 1
                    conv_downsamples.append(DWConv(c_in, c_out, stride=2, mid_relu=False, last_relu=False))
                    fuse_layers[-1].append(nn.Sequential(*conv_downsamples))
        return fuse_layers

    def forward(self, x):
        if self.in_branches == 1:
            return [self.layers[0](x[0])]
        
        out = self.layers(x)

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                # TODO : 这里 我把 j 改成从 1 开始，而不是0，否则会会加两次out[0] 但是模型的准确率会下降
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.in_branches):
                    if i == j:
                        y += out[j]
                    else:
                        y += self.fuse_layers[i][j](out[j])
                out_fuse.append(self.relu(y))
        out = out_fuse if self.in_branches > 1 else [out_fuse[0]]
        return out

class StemModule(nn.Module):
    def __init__(self, in_channels, stem_channels, out_channels, expand_ratio):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, 2, 1),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU()
        )
        self.out_channels = out_channels
        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == out_channels:
            inc_channels = out_channels - branch_channels
        else:
            inc_channels = out_channels - stem_channels
        
        self.branch1 = DWConv(branch_channels, inc_channels, stride=2, mid_relu=False, bias=True)

        self.expand_conv = nn.Sequential(
            nn.Conv2d(branch_channels, mid_channels, 1, 1, 0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1, groups=mid_channels),
            nn.BatchNorm2d(mid_channels)
        )
        self.linear_conv = nn.Sequential(
            nn.Conv2d(mid_channels, branch_channels if stem_channels == out_channels else stem_channels, 1, 1, 0),
            nn.BatchNorm2d(branch_channels if stem_channels == out_channels else stem_channels),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)

        x2 = self.expand_conv(x2)
        x2 = self.depthwise_conv(x2)
        x2 = self.linear_conv(x2)

        out = torch.cat((self.branch1(x1), x2), dim=1)
        out = channel_shuffle(out, 2)
        return out

class IterativeHead(nn.Module):
    """
      从分辨率最低的最后一个分支开始处理，逐次把小分辨率分支结果用双线性插值的方式叠加到次小分辨率结果上， 在进过深度可分离卷积进行特征提取，得到新的次小分辨率结果。
    """
    def __init__(self, in_channels):
        super().__init__()
        projects = []
        num_branches = len(in_channels)
        self.in_channels = in_channels[::-1]

        for i in range(num_branches):
            if i != num_branches -1:
                projects.append(DWConv(self.in_channels[i], self.in_channels[i+1]))
            else:
                projects.append(DWConv(self.in_channels[i], self.in_channels[i]))
        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]
        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x, size=s.shape[-2:], mode='bilinear', align_corners=True
                )
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s
        return y[::-1]

class LiteHRNet(nn.Module):
    # def __init__(self, cfg):
    def __init__(self, cfg):
        super().__init__()
        out_channel= cfg.MODEL.get('output_channel', cfg.DATASET.num_joints)

        self.stem = StemModule(
            in_channels=3, stem_channels=32, 
            out_channels=32, expand_ratio=1
        )
        self.num_stages = 3
        self.with_head = True
        self.stages_spec = dict(
                num_modules=(3, 8, 3),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                ))     
        num_channels_last = [self.stem.out_channels]

        for i in range(self.num_stages):
            num_channels = self.stages_spec['num_channels'][i]
            num_channels = [num_channels[i] for i in range(len(num_channels))]
            setattr(
                self, 'transition{}'.format(i), self._make_transition_layer(num_channels_last, num_channels)
            )
            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels)
            setattr(self, 'stage{}'.format(i), stage)
        
        if self.with_head:
            self.head_layer = IterativeHead(in_channels=num_channels_last)
        self.out_conv = nn.Conv2d(40, out_channel, 1, 1, 0)  # 需要自己加一个检测头
    
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """将上一个stage的输出进行处理得到下一个stage的输入, 对于同一层次，如果通道数不同则变通道，如果为新层次，则下采样"""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)  # 前一个stage通道数

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    c_in = num_channels_pre_layer[i]
                    c_out = num_channels_cur_layer[i]
                    transition_layers.append(DWConv(c_in, c_out, mid_relu=False))
                else:
                    transition_layers.append(None)
            else:
                conv_downsample = []
                for j in range(i + 1 - num_branches_pre):
                    c_in = num_channels_pre_layer[-1]
                    # 判断当前分支是否刚好为新分支
                    c_out = num_channels_cur_layer[i] if j == i - num_branches_pre else c_in
                    conv_downsample.append(DWConv(c_in, c_out, stride=2, mid_relu=False))
                transition_layers.append(nn.Sequential(*conv_downsample))
        return nn.ModuleList(transition_layers)
                    
    def _make_stage(self, stages_spec, stage_index, in_channels):
        num_modules = stages_spec['num_modules'][stage_index]
        num_branches = stages_spec['num_branches'][stage_index]
        num_blocks = stages_spec['num_blocks'][stage_index]
        reduce_ratio = stages_spec['reduce_ratios'][stage_index]
        with_fuse = stages_spec['with_fuse'][stage_index]
        
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            modules.append(
                StageModule(num_branches, num_blocks, in_channels, reduce_ratio, with_fuse)
            )
            in_channels = modules[-1].in_channels
        return nn.Sequential(*modules), in_channels

    def forward(self, x):
        x = self.stem(x)
        y_list = [x]

        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, f'transition{i}')
            for j in range(self.stages_spec['num_branches'][i]):
                if transition[j]:
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, f'stage{i}')(x_list)

        x = y_list
        if self.with_head:
            x = self.head_layer(x)
        # return [x[0]]
        out = self.out_conv(x[0])
        return out



