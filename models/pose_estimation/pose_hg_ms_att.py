
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from torch.nn import functional as F

# class Merge(nn.Module):
#     def __init__(self, x_dim, y_dim):
#         super(Merge, self).__init__()
#         self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

#     def forward(self, x):
#         return self.conv(x)

# class Conv(nn.Module):
#     def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
#         super(Conv, self).__init__()
#         self.inp_dim = inp_dim
#         self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
#         self.relu = None
#         self.bn = None
#         if relu:
#             self.relu = nn.ReLU()
#         if bn:
#             self.bn = nn.BatchNorm2d(out_dim)

#     def forward(self, x):
#         # assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

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

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
   
        relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(inp_dim),
            relu,
            nn.Conv2d(inp_dim, int(out_dim / 2), 1, 1, 0),
            nn.BatchNorm2d(int(out_dim / 2)),
            relu,
            nn.Conv2d(int(out_dim / 2), int(out_dim / 2), 3, 1, 1),
            nn.BatchNorm2d(int(out_dim / 2)),  
            relu,
            nn.Conv2d(int(out_dim / 2), out_dim, 1, 1, 0), 
        )
        
        if inp_dim == out_dim:
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = nn.Conv2d(inp_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        return self.skip_layer(x) + self.conv(x)

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
    
class MultiScaleAttentionBlock(nn.Module):
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
    
class MultiScaleAttentionHourglass(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.check_init(cfg)
        self.nstack = len(cfg['hm_loss_factor'])
        inp_dim=cfg['main_channels']
        increase=cfg['increase']
        oup_dim = cfg['n_joints'] + 3 if cfg['with_region_map'] else cfg['n_joints']
        
        self.pre = my_pelee_stem(inp_dim)
        
        self.hgs = nn.ModuleList([
            Hourglass(cfg['hg_depth'], inp_dim,
                        increase, basic_block=MultiScaleAttentionBlock)
            for _ in range(self.nstack)
            ])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                nn.BatchNorm2d(inp_dim),
                nn.ReLU(),
                nn.Conv2d(inp_dim, inp_dim, 1, 1, 0)
            ) for _ in range(self.nstack)])

        self.outs = nn.ModuleList([
                nn.Conv2d(inp_dim, oup_dim, 1, 1, 0)
            for _ in range(self.nstack)])
        
        self.merge_features = nn.ModuleList([
                nn.Conv2d(inp_dim, inp_dim, 1, 1, 0)
            for _ in range(self.nstack - 1)])
        
        self.merge_preds = nn.ModuleList([
                nn.Conv2d(oup_dim, inp_dim, 1, 1, 0)
            for _ in range(self.nstack - 1)])
        
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
        kpts = hm_preds[-1][:, :-3]
        # if imgs.shape[-1] != self.image_size[0]:
        #     kpts = F.interpolate(kpts , scale_factor=2, mode='nearest')
        kpts = rearrange(kpts, 'b c h w -> b c (h w)')
        pred_x = self.pred_x(kpts)  # (b, c, w * k)
        pred_y = self.pred_y(kpts)  # (b, c, h * k)   
        return hm_preds, pred_x, pred_y
    
    def check_init(self, cfg):
        assert isinstance(cfg['hm_size'], (tuple, list)), \
            "hm_size should be a tuple or list"
        assert isinstance(cfg['hm_sigma'], (tuple, list)), \
            "hm_sigma should be a tuple or list"
        assert isinstance(cfg['hm_loss_factor'], (tuple, list)), \
            "loss_factor should be a tuple or list"
        assert len(cfg['hm_size']) == len(cfg['hm_sigma']), "Length must be equal !"
        assert len(cfg['hm_size']) == len(cfg['hm_loss_factor']), "Length must be equal !"
        

        
