from torch import nn
import torch.nn.functional as F
from .deconv_head import DeconvHead

class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1,
                 padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride,
                      padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(True)
        )
    def forward(self, x):
        return self.conv(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, 1, 0),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, stride, 1),
            nn.BatchNorm2d(in_channels // 4),  
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, out_channels, 1, 1, 0), 
            nn.BatchNorm2d(out_channels),  
        )
        self.downsample = downsample if downsample != None else nn.Identity()
    def forward(self, x):
        return F.relu(self.downsample(x) + self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),  
        )
        self.downsample = downsample if downsample != None else nn.Identity()
    def forward(self, x):
        return F.relu(self.downsample(x) + self.conv(x))

class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        stride (int): stride of the first block. Default: 1.
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """
    def __init__(self, block, num_blocks, in_channels, out_channels,
                 stride=1, downsample_first=True):
        
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        if downsample_first:  # for ResNet
            layers.append(block(in_channels, out_channels, stride, downsample))
            for _ in range(1, num_blocks):
                layers.append(block(out_channels, out_channels, 1))
        else:  # for Hourglass Module
            for _ in range(0, num_blocks-1):
                layers.append(block(in_channels, in_channels, 1))
            layers.append(block(in_channels, out_channels, stride, downsample))
        
        super().__init__(*layers)


class PoseResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    def __init__(self, cfg=None):
        super().__init__()
        self.depth = cfg.MODEL.get('depth', 50)
        if self.depth not in self.arch_settings:
            raise KeyError(f'invalid depth {self.depth} for resnet')

        self.head_out_channels = cfg.MODEL.get('output_channel', 21)
        self.out_indices = cfg.MODEL.get('out_indices', (3, ))  # (0, 1, 2, 3)
        self.stem_channels = cfg.MODEL.get('stem_channels', 64)
        self.base_channels = cfg.MODEL.get('base_channels', 64)
        self.stride = cfg.MODEL.get('strides', (1, 2, 2, 2))
        # Replace 7x7 conv in input stem with 3 3x3 conv.
        self.deep_stem = cfg.MODEL.get('deep_stem', False)
        self.num_stages = cfg.MODEL.get('num_stages', 4)
        assert 1 <= self.num_stages <= 4
        self.block, stage_blocks = self.arch_settings[self.depth]
        self.stage_blocks = stage_blocks[:self.num_stages]
        
        self._make_stem_layer()
        res_layers = []
        _in_channels = self.stem_channels
        _out_channels = self.base_channels * self._get_expansion(self.block)
        for i, num_blocks in enumerate(self.stage_blocks):
            res_layers.append(
                ResLayer(self.block, num_blocks, _in_channels,
                         _out_channels,self.stride[i])
                )
            _in_channels = _out_channels
            _out_channels *= 2
        self.res_layers = nn.ModuleList(res_layers)
        
        self.out_head = DeconvHead(
            in_channels=_out_channels // 2,  # 最后多乘了2
            out_channels=self.head_out_channels,
            num_deconv_layers=3,               # 三个deconv进行上采样
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, 4),
            extra=dict(
                final_conv_kernel=1,
                # 最终输出层前的额外卷积层
                num_conv_layers=0,  # 最终输出头只需要一个卷积层即可, 不需要额外卷积
                num_conv_kernels=[]
            )
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        outs = []
        for i in range(self.num_stages):
            x = self.res_layers[i](x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return self.out_head(outs[0])
        return tuple([self.out_head[y] for y in outs])


    def _make_stem_layer(self):
        if self.deep_stem:
            self.stem = nn.Sequential(
                CBL(3, self.stem_channels // 2, 3, 2, 1),
                CBL(self.stem_channels // 2, self.stem_channels // 2, 3, 1, 1),
                CBL(self.stem_channels // 2, self.stem_channels, 3, 1, 1)
            )
        else:
            self.stem = CBL(3, self.stem_channels, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def _get_expansion(self, block):
        if issubclass(block, BasicBlock):
            return 1
        elif issubclass(block, Bottleneck):
            return 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
            
            