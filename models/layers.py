from torch import nn
import torch
from models.attention import SELayer, SoftPooling


class ConvBnReLu(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, bias=False):
        super(ConvBnReLu, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(k, k), stride=(s, s), padding=(p, p), bias=bias),
            nn.BatchNorm2d(c_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.cbr(x)


class SoftmaxHeatmap(nn.Module):
    """ 我自己想的一个输出层，输出热图中每个点的概率"""

    def __init__(self, in_channels, out_channels):
        super(SoftmaxHeatmap, self).__init__()
        self.softmax_row = nn.Softmax(dim=-1)
        self.softmax_col = nn.Softmax(dim=-2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        # x.shape = (batch, n_joints, h, w)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out_row = self.softmax_row(out1)
        out_col = self.softmax_row(out2)
        return (out_row + out_col) / 2.0


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP
        https://zhuanlan.zhihu.com/p/396724233
    """

    def __init__(
            self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1)
        self.activation1 = nn.SiLU()
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = nn.Conv2d(conv2_channels, out_channels, 1, 1)
        self.activation2 = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        x = self.activation2(x)
        return x


class SPP_layer(nn.Module):
    def __init__(self, c_in, c_out, kernel_sizes=(5, 9, 13)):
        super(SPP_layer, self).__init__()
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        self.cbr = nn.Sequential(
            nn.Conv2d(c_in * 4, c_out, 1, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        # input:(batch, c, h, w) --> output:(batch, 4*c, h, w)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.cbr(x)
        return x


# --------- SRHandNet -----------

class Conv(nn.Module):
    def __init__(self, in_dim, out_dim,
                 kernel_size=3, stride=1,
                 bn=False, relu=True):
        super(Conv, self).__init__()
        self.in_dim = in_dim
        bias = True if not bn else False
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                              kernel_size=(kernel_size, kernel_size), stride=(stride, stride),
                              padding=(kernel_size - 1) // 2, bias=bias)  # todo: padding value?
        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(num_features=out_dim) if bn else None

    def forward(self, x):
        # x -> (batch, c, h, w)
        # assert x.shape[1] == self.in_dim, "{} != {}".format(x.shape[1], self.in_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Residual, self).__init__()
        if out_dim == 1:
            raise ValueError
        mid_dim = out_dim // 2  # raise error if out_dim <= 1
        self.ConvBlock = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, mid_dim, 1, 1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(),
            nn.Conv2d(mid_dim, out_dim, 1, 1),
        )

        if in_dim == out_dim:
            self.skip_layer = nn.Identity()  # 不做处理，直接相加
        else: 
            self.skip_layer = Conv(in_dim, out_dim, 1, relu=False)

    def forward(self, x):
        residual = self.skip_layer(x)

        out = self.ConvBlock(x)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, double=False, increment=0, pool_type='max_pool'):
        super(Hourglass, self).__init__()
        if double:
            nf = f * 2
        else:
            nf = f + increment
        self.skip_block = Residual(f, f)

        # lower branch
        pool_list = ['max_pool', 'soft_pool', 'avg_pool']
        if pool_type in pool_list:
            if pool_type == pool_list[0]:
                self.pool = nn.MaxPool2d(2, 2)  #
            elif pool_type == pool_list[1]:
                self.pool = SoftPooling(2, 2)  # todo 在Linux上面跑要用原作者的实现，运行速度更快
            elif pool_type == pool_list[2]:
                self.pool = nn.AvgPool2d(2, 2)
            else:
                raise NotImplementedError('Pool:{}is not in {} !'.format(pool_type, pool_list))

        self.low_residual = Residual(f, nf)
        self.n = n

        # Recursive hourglass
        if n > 1:
            self.mid_block = Hourglass(n - 1, nf)
        else:
            self.mid_block = Residual(nf, nf)
        self.up_residual = Residual(nf, f)
        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        up1 = self.skip_block(x)

        pool = self.pool(x)

        low1 = self.low_residual(pool)

        mid_out = self.mid_block(low1)

        low2 = self.up_residual(mid_out)

        up2 = self.up_sample(low2)
        # print(f"{up1.shape=}"), print(f"{up2.shape=}")
        return up1 + up2


# ----------- HRNet modules ----------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=(1, 1), bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        mid_channel = int(out_channel / self.expansion)

        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel, momentum=bn_momentum)

        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=(3, 3),
                               stride=stride, padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel, momentum=bn_momentum)

        self.conv3 = nn.Conv2d(mid_channel, out_channel, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=(1, 1), downsample=False, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel, momentum=bn_momentum)

        self.downsample = nn.Conv2d(in_channel, out_channel, 3, stride, 1) if downsample else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# -------------------------------------

class BulatBlock(nn.Module):
    """
        soft gated skip connection
        Paper: 'Toward fast and accurate human pose estimation via soft-gated skip connections'
        source code:  https://github.com/benjs/hourglass_networks
        Bulat is the name of the author 
    """
    def __init__(self, c_in, c_out):
        super().__init__()

        assert (c_in // 4) != 0, f'ERROR: c_in // 4 == 0 , {c_in=}'

        if c_in != c_out:
            self.skip_connection = nn.Conv2d(c_in, c_out, 1, 1, 0)
        else:
            self.skip_connection = nn.Identity()

        # mid layer
        self.brc1 = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.ReLU(True),
            nn.Conv2d(c_in, c_in // 2, 3, 1, 1, bias=False))
        
        self.brc2 = nn.Sequential(
            nn.BatchNorm2d(c_in // 2),
            nn.ReLU(True),
            nn.Conv2d(c_in // 2, c_in // 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_in // 2),
            nn.ReLU(True),
            nn.Conv2d(c_in // 2, c_in // 2, 3, 1, 1, bias=False))
                
        self.conv1x1 =  nn.Conv2d(c_in, c_out, 1, 1, 0)
        self.senet = SELayer(c_out)
            
    def forward(self, x):
        # main branch
        out1 = self.brc1(x)
        out2 = self.brc2(out1)

        # skip connection:
        skip_feature = self.skip_connection(x)
        main_feature = torch.cat([out1, out2], dim=1)
        main_feature = self.conv1x1(main_feature)
        
        # feature fusion
        out = main_feature + skip_feature
        out = self.senet(out)
        return out


class Hourglass_Bulat(nn.Module):
    def __init__(self, n, f, double=False, increment=0, pool_type='max_pool'):
        super(Hourglass_Bulat, self).__init__()
        if double:
            nf = f * 2
        else:
            nf = f + increment
        self.skip_block = BulatBlock(f, f)

        # lower branch
        pool_list = ['max_pool', 'soft_pool', 'avg_pool']
        if pool_type in pool_list:
            if pool_type == pool_list[0]:
                self.pool = nn.MaxPool2d(2, 2)  #
            elif pool_type == pool_list[1]:
                self.pool = SoftPooling(2, 2)  # todo 在Linux上面跑要用原作者的实现，运行速度更快
            elif pool_type == pool_list[2]:
                self.pool = nn.AvgPool2d(2, 2)
            else:
                raise NotImplementedError('Pool:{}is not in {} !'.format(pool_type, pool_list))

        self.low_residual = BulatBlock(f, nf)
        self.n = n

        # Recursive hourglass
        if n > 1:
            self.mid_block = Hourglass(n - 1, nf)
        else:
            self.mid_block = BulatBlock(nf, nf)
        self.up_residual = BulatBlock(nf, f)
        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        up1 = self.skip_block(x)

        pool = self.pool(x)

        low1 = self.low_residual(pool)

        mid_out = self.mid_block(low1)

        low2 = self.up_residual(mid_out)

        up2 = self.up_sample(low2)
        # print(f"{up1.shape=}"), print(f"{up2.shape=}")
        return up1 + up2

# lite hrnet -------------------------------------------

class DWConv(nn.Module):
    """DepthwiseSeparableConvModul 深度可分离卷积"""
    def __init__(self, in_channel, out_channel, stride=1, mid_relu=True, last_relu=True, bias=False, dilation=1, padding=1):
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


def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

class DWConvBlock(nn.Module):
    """这个是LiteHrnet中基础模块的简化版"""
    def __init__(self, in_c, out_c, stride=1, mid_relu=True, last_relu=True, bias=False):
        super().__init__()
        self.conv1x1 = ConvBnReLu(in_c, out_c, 1, 1, 0)
        self.depth_separate_conv = DWConv(out_c // 2, out_c //2, stride, mid_relu, last_relu, bias)
    
    def forward(self, x):
        x = self.conv1x1(x)
        x1, x2 = x.chunk(2, dim=1) 
        x2 = self.depth_separate_conv(x2)
        x = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(x, 2)
        return out
    
    
    
    
    