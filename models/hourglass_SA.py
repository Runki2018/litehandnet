from threading import main_thread
from pyparsing import TokenConverter
from pyrsistent import inc
import torch
from torch import nn
from torch.nn import functional as F
from utils.training_kits import load_pretrained_state
from config.config import config_dict as cfg
from models.attention import SELayer, NAM_Channel_Att
from models.layers import DWConv, LiteHG, fuse_block
from einops import rearrange, repeat
from torch.nn import functional as F

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        # assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)    
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        
        if inp_dim == out_dim:
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)

    def forward(self, x):
        # if self.need_skip:
        #     residual = self.skip_layer(x)
        # else:
        #     residual = x
        residual = self.skip_layer(x)
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

class BRC(nn.Module):
    """  BN + Relu + Conv2d """    
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, dilation=1):
        super(BRC, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride,
                            padding=padding, bias=bias, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(inp_dim)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

class Hourglass(nn.Module):
    def __init__(self, n, f, increase=0, basic_block=Residual):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = basic_block(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = basic_block(f, nf)
        # Recursive hourglass
        if n > 1:
            self.low2 = Hourglass(n - 1, nf)
        else:
            self.low2 = basic_block(nf, nf)
        self.low3 = basic_block(nf, f)
        # self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)

        low2 = self.low2(low1)

        low3 = self.low3(low2)
        # up2 = self.up2(low3)
        up2 = F.interpolate(low3, scale_factor=2)
        return up1 + up2

class MSRB_D_DWConv(nn.Module):
    """
    https://blog.csdn.net/KevinZ5111/article/details/104730835?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-4-104730835.pc_agg_new_rank&utm_term=block%E6%94%B9%E8%BF%9B+residual&spm=1000.2123.3001.4430
    """
    def __init__(self, in_c, out_c):
        super(MSRB_D_DWConv, self).__init__()


        mid_c = in_c // 2
        self.conv1 = BRC(in_c, mid_c, 1, 1, 0)

        self.mid1_conv = nn.ModuleList([
            nn.Sequential(
                DWConv(mid_c, mid_c // 2),
                DWConv(mid_c // 2, mid_c // 2)), 
            nn.Sequential(
                DWConv(mid_c, mid_c),
                DWConv(mid_c, mid_c),        
            )])
        
        self.mid2_conv = nn.ModuleList([
            DWConv(mid_c, mid_c // 2, dilation=2, padding=2),
            DWConv(mid_c, mid_c, dilation=2, padding=2)])
            
        self.conv2 = BRC(2 * mid_c, in_c, 1, 1, 0, bias=False)
        self.conv3 = BRC(in_c, out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        m = self.conv1(x)
        for i in range(2):
            m1 = self.mid1_conv[i](m)
            m2 = self.mid2_conv[i](m)
            m = torch.cat([m1, m2], dim=1)

        features = self.conv2(m) + x
        out = self.conv3(features)
        return out

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
        # self.conv2 = BRC(2 * mid_c, in_c, 1, 1, 0, bias=False)
        self.conv3 = BRC(in_c, out_c, 1, 1, 0, bias=False)
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
        # self.nam_channel_attention = NAM_Channel_Att(channels=out_c)

    def forward(self, x):
        m = self.conv1(x)
        for i in range(2):
            m1 = self.mid1_conv[i](m)
            m2 = self.mid2_conv[i](m)
            m = torch.cat([m1, m2], dim=1)

        # features = self.conv2(m) + x
        features = m + x
        out = self.conv3(features)
        b, c, _, _ = out.shape
        out = out * self.att(out).view(b, c, 1, 1)
        # out = self.nam_channel_attention(out)
        return out

class my_pelee_stem(nn.Module):
    """ 我在Conv1中再加了一个3x3卷积，来提高stem的初始感受野"""
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
        
        self.conv1x1 = nn.Sequential(   
            nn.Conv2d(mid_channel * 2, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.conv1(x)
        b1 = self.branch1(out)
        b2 = self.branch2(out)
        out = torch.cat([b1, b2], dim=1)
        out = self.conv1x1(out)
        return out
    
class HourglassNet_SA(nn.Module):
    def __init__(self, nstack=1, inp_dim=128, oup_dim=24,  increase=0,
        basic_block=ME_att):
        super().__init__()

        self.pre = my_pelee_stem(128)

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, increase, basic_block=basic_block),
            ) for _ in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for _ in range(nstack)])

        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for _ in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for _ in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for _ in range(nstack - 1)])
        self.nstack = nstack
        
        self.image_size = cfg['image_size']  # (w, h)
        k = cfg['simdr_split_ratio']  # default k = 2 
        in_features = int(self.image_size[0] * self.image_size[1] / (4 ** 2))  # 下采样率是4，所以除以16  
        self.pred_x = nn.Linear(in_features, int(self.image_size[0] * k)) 
        self.pred_y = nn.Linear(in_features, int(self.image_size[1] * k))
        
    def forward(self, imgs):
        # our posenet
        x = self.pre(imgs)

        hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)

            feature = self.features[i](hg)
            preds = self.outs[i](feature)

            hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
           
        # predict keypoints
        kpts = hm_preds[-1][:, 3:]
        if imgs.shape[-1] != self.image_size[0]:
            kpts = F.interpolate(kpts , scale_factor=2, mode='nearest')
        kpts = rearrange(kpts, 'b c h w -> b c (h w)')
        pred_x = self.pred_x(kpts)  # (b, c, w * k)
        pred_y = self.pred_y(kpts)  # (b, c, h * k)   
        return hm_preds, pred_x, pred_y

        
