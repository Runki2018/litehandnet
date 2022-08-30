import torch
from torch import nn
from torch.nn import functional as F


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % groups == 0)
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class ChannelAttension(nn.Module):
    def __init__(self, channel, deploy=False):
        super().__init__()
        self.deploy = deploy
        if self.deploy:
            self.rbr_reparam = nn.Conv2d(channel, channel, 3, 1, 0, groups=channel)
        else:
            self.conv3x3 = nn.Sequential()
            self.conv3x3.add_module('conv', nn.Conv2d(channel, channel, 3, 1, 0,
                                                groups=channel, bias=False))
            self.conv3x3.add_module('bn', nn.BatchNorm2d(channel))
        self.conv1x1 = nn.Sequential(
            nn.Dropout2d(p=0.3),
            nn.Conv2d(channel, channel // 2, 1, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, 1, 1, 0),
            nn.Sigmoid()
            # nn.Hardtanh()
        )
 
    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, (3, 3))
        if hasattr(self, 'rbr_reparam'):
            att = self.rbr_reparam(y)
        else:
            att = self.conv3x3(y)
        return x * self.conv1x1(att)

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return   
        kernel = self.conv3x3.conv.weight
        running_mean = self.conv3x3.bn.running_mean
        running_var = self.conv3x3.bn.running_var
        gamma = self.conv3x3.bn.weight
        beta = self.conv3x3.bn.bias
        eps = self.conv3x3.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        kernel = kernel * t
        bias = beta - running_mean * gamma / std
        self.rbr_reparam = nn.Conv2d(self.conv3x3.conv.in_channels,
                                     self.conv3x3.conv.in_channels,
                                     3, 1, 0, 
                                     groups=self.conv3x3.conv.in_channels)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv3x3')
        self.deploy = True