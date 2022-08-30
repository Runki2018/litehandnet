from torch import nn
from models.weight_init import kaiming_init, constant_init
from .deconv_head import DeconvHead


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number down to the nearest value that can
    be divisible by the divisor.

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int, optional): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float, optional): The minimum ratio of the rounded channel
            number to the original channel number. Default: 0.9.
    Returns:
        int: The modified output channel number
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value

class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1,
                 padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride,
                      padding, dilation, groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )
    def forward(self, x):
        return self.conv(x)


class InvertedResidual(nn.Module):
    """InvertedResidual block for MobileNetV2
      Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        
        layers = []
        if expand_ratio != 1:
            layers.append(CBL(in_channels, hidden_dim))
        layers.extend([
            CBL(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim),
            CBL(hidden_dim, out_channels)
        ])
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class PoseMobileNetV2(nn.Module):
    """MobileNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (7, ).
    """
    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks, stride.
    arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
                     [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
                     [6, 320, 1, 1]]
    def __init__(self, cfg):
        super().__init__()
        self.widen_factor = cfg.MODEL.get('widen_factor', 1)
        self.out_indices = cfg.MODEL.get('out_indices', (7,))
        self.head_out_channels = cfg.MODEL.get('output_channel', 21)
        for index in self.out_indices:
            if index not in range(0, 8):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 8). But received {index}')
        
        self.in_channels = make_divisible(32 * self.widen_factor, 8)
        self.conv1 = CBL(3, self.in_channels, 3, 2, 1)
        
        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            out_channels = make_divisible(channel * self.widen_factor, 8)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio
            )
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)
        
        if self.widen_factor > 1.0:
            self.out_channels = int(1280 * self.widen_factor)
        else:
            self.out_channels = 1280
        
        layer = CBL(self.in_channels, self.out_channels)
        self.add_module('conv2', layer)   # [B, 1280*self.widen_factor, H/4, W/4]
        self.layers.append('conv2')

        self.out_head = DeconvHead(
            in_channels=self.out_channels,
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


    def make_layer(self, out_channels, num_blocks, stride, expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(
                InvertedResidual(
                    self.in_channels,
                    out_channels,
                    stride,
                    expand_ratio=expand_ratio,
                   )
                )
            self.in_channels = out_channels

        return nn.Sequential(*layers)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)


    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        # if len(outs) == 1:
        #     return outs[0]
        # return tuple(outs)
        
        if len(outs) == 1:
            return self.out_head(outs[0])
        return tuple([self.out_head(a) for a in outs])

        

        

    