import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .common import SEBlock, ChannelAttension, channel_shuffle


def conv_bn(in_channels, out_channels, kernel_size, stride, padding,
            dilation=1, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1,
                 padding=0, dilation=1, groups=1, deploy=False,
                 activation=nn.LeakyReLU, inplace=False):
        super().__init__()
        self.deploy = deploy
        if activation != None:
            self.nonlinearity = activation(inplace)
        else:
            self.nonlinearity = nn.Identity()

        if self.deploy:
            self.rep_conv = nn.Conv2d(in_channels, out_channels, kernel,
                                         stride, padding, dilation, groups, bias=True)
        else:
            self.conv = conv_bn(in_channels, out_channels, kernel,
                                   stride, padding, dilation, groups)

    def forward(self, x):
        if hasattr(self, 'rep_conv'):
            return self.nonlinearity(self.rep_conv(x))
        return self.nonlinearity(self.conv(x))

    def switch_to_deploy(self):
        if hasattr(self, 'rep_conv'):
            return   
        kernel = self.conv.conv.weight
        running_mean = self.conv.bn.running_mean
        running_var = self.conv.bn.running_var
        gamma = self.conv.bn.weight
        beta = self.conv.bn.bias
        eps = self.conv.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        kernel = kernel * t
        bias = beta - running_mean * gamma / std
        self.rep_conv = nn.Conv2d(self.conv.conv.in_channels,
                                     self.conv.conv.in_channels,
                                     self.conv.conv.kernel_size,
                                     self.conv.conv.stride,
                                     self.conv.conv.padding,
                                     self.conv.conv.dilation,
                                     self.conv.conv.groups,
                                     bias=True
                                     )
        self.rep_conv.weight.data = kernel
        self.rep_conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.deploy = True


class RepBlock(nn.Module):
    """https://github.com/DingXiaoH/RepVGG
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, padding_mode='zeros',
                 deploy=False, ca_type=None, activation=nn.LeakyReLU, inplace=False,
                 identity=True):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # self.kernel_receptive_field = kernel_size + (kernel_size - 1)*(dilation-1)
        if activation != None:
            self.nonlinearity = activation(inplace=inplace)
        else:
            self.nonlinearity = nn.Identity()

        if ca_type == 'se':
            self.ca = SEBlock(out_channels, internal_neurons=out_channels // 16)
        elif ca_type == 'ca':
            self.ca = ChannelAttension(out_channels)
        else:
            self.ca = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, 
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)
        else:
            if identity and out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)
            else:
                self.rbr_identity = None

            self.rbr_dense = conv_bn(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups)

            self.rbr_1x1 = conv_bn(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                #    padding=(padding - kernel_size // 2),
                                   padding=0,
                                   dilation=1,
                                   groups=groups)
            # print('RepBlock, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.ca(self.rbr_reparam(inputs)))
        out1 = self.rbr_dense(inputs)
        out2 = self.rbr_1x1(inputs)
        out3 = 0 if self.rbr_identity is None else self.rbr_identity(inputs)
        # use dropout to increase the independence of each branch, model embedding
        # 我觉得可以增加各个分支的独立性，加强子模型训练的效果，增加子模型的多样性。
        return self.nonlinearity(self.ca(out1 + out2 + out3))

    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel, bias = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        
        kernel += self._pad_1x1_to_kxk_tensor(kernel1x1) + kernelid
        bias += bias1x1 + biasid
        return kernel, bias 

    def _pad_1x1_to_kxk_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            pad = self.kernel_size // 2
            return torch.nn.functional.pad(kernel1x1, [pad,pad,pad,pad])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                k = self.kernel_size
                kernel_value = np.zeros((self.in_channels, input_dim, k, k), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, k//2, k//2] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size,
                                     stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding,
                                     dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups,
                                     bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

# -------------------------------------------------------------------------------------
class RepBasicUnit(nn.Module):
    """
    from shffulenetv2
    https://github.com/Randl/ShuffleNetV2-pytorch/blob/master/model.py#L63
    """
    def __init__(self, inplanes, outplanes, c_tag=0.5, ca_type='none',
                 residual=False, groups=2):
        super(RepBasicUnit, self).__init__()
        self.left_part = round(c_tag * inplanes)
        self.right_part_in = inplanes - self.left_part
        self.right_part_out = outplanes - self.left_part
        
        self.conv = nn.Sequential(
            RepConv(self.right_part_in, self.right_part_out, kernel=1, activation=nn.ReLU),
            RepConv(self.right_part_out, self.right_part_out, kernel=3, padding=1,
                    groups=self.right_part_out, activation=None),
            RepConv(self.right_part_out, self.right_part_out, kernel=1,
                    ca_type=ca_type, activation=nn.ReLU)
        )

        self.inplanes = inplanes
        self.outplanes = outplanes
        self.residual = residual
        self.groups = groups

    def forward(self, x):
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        out = self.conv(right)

        if self.residual and self.inplanes == self.outplanes:
            out += right

        return channel_shuffle(torch.cat((left, out), 1), self.groups)

class RepDownsampleUnit(nn.Module):
    """
    from shffulenetv2
    https://github.com/Randl/ShuffleNetV2-pytorch/blob/master/model.py#L63
    """
    def __init__(self, inplanes, activation=nn.ReLU, groups=2):
        super(RepDownsampleUnit, self).__init__()
        self.conv_right = nn.Sequential(
            RepConv(inplanes, inplanes, 1, 1, 0, activation=activation),
            RepConv(inplanes, inplanes, 3, 2, 1, groups=inplanes, activation=None),
            RepConv(inplanes, inplanes, 1, 1, 0, activation=activation)
        )
        self.conv_left = nn.Sequential(
            RepConv(inplanes, inplanes, 3, 2, 1, groups=inplanes, activation=None),
            RepConv(inplanes, inplanes, 1, 1, 0, activation=activation)
        )
        self.groups = groups

    def forward(self, x):
        out_r = self.conv_right(x)
        out_l = self.conv_left(x)
        return channel_shuffle(torch.cat((out_r, out_l), 1), self.groups)

# -------------------------------------------------------------------------------------
