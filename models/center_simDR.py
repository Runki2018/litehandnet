from re import S
import torch
from torch import nn
from utils.training_kits import load_pretrained_state
from config.config import config_dict as cfg
from models.attention import Flatten, SELayer, NAM_Channel_Att
from models.layers import DWConv, SplitDWConv
from einops import rearrange, repeat

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
                DWConv(mid_c // 2, mid_c // 2)),
            nn.Sequential(
                DWConv(mid_c, mid_c, dilation=2, padding=2),
                DWConv(mid_c, mid_c))
            ])
        
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

    def forward(self, x):
        m = self.conv1(x)
        for i in range(2):
            m1 = self.mid1_conv[i](m)
            m2 = self.mid2_conv[i](m)
            m = torch.cat([m1, m2], dim=1)

        features = m + x
        out = self.conv3(features)
        b, c, _, _ = out.shape
        out = out * self.att(out).view(b, c, 1, 1)
        return out

class Hourglass_module(nn.Module):
    def __init__(self, n, channels):
        super(Hourglass_module, self).__init__()

        assert n == len(channels), 'Error: num of layer != num of channels'
        self.n_stack = n
        
        self.skip_layer = nn.ModuleList([])
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        # TODO： 简化skip 和 最小层度的卷积层 是否会精度下降？    
        for i in range(n):
            self.skip_layer.append(SplitDWConv(channels[i], channels[i]))  
            if i != n-1:
                self.encoder.append(ME_att(channels[i], channels[i+1]))
                self.decoder.append(ME_att(channels[i+1], channels[i]))        
            else:
                self.encoder.append(SplitDWConv(channels[i], channels[i]))
                self.decoder.append(SplitDWConv(channels[i], channels[i]))  

        self.decoder = self.decoder[::-1]
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        out_skip = []
        # encoder
        for i in range(self.n_stack):
            out_skip.append(self.skip_layer[i](x))
            if i != self.n_stack -1:
                x = self.pool(x)
            x = self.encoder[i](x)
        # decoder
        for i in range(self.n_stack):
            x = self.decoder[i](x)
            x = x + out_skip.pop()
            if i != self.n_stack -1:
                x = self.upsample(x)
        return x

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

class LiteHourglassNet(nn.Module):
    def __init__(self):
        super().__init__()
        nstack = cfg['nstack']
        channels = cfg['hg_channels']
        out_kernel = cfg['out_kernel']
        image_size = cfg['image_size']  # (w, h)
        k = cfg['simdr_split_ratio']  # default k = 2 
        assert out_kernel in [1, 3], 'Error: kernel size not in [1, 3]'
        inp_dim = channels[0]
        oup_dim = cfg['n_joints']

        self.nstack = nstack
        self.pre = my_pelee_stem(out_channel=inp_dim)

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass_module(len(channels), channels),
            ) for _ in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                SplitDWConv(inp_dim, inp_dim),
                BRC(inp_dim, inp_dim),
            ) for _ in range(nstack)])
        
        # 去除了中间监督和简化了融合模块
        self.merge = nn.ModuleList([BRC(inp_dim * 2, inp_dim, 1, 1, 0) for _ in range(nstack)]) 

        # todo : center 需要一类一个通道吗？
        # TODO: 根据训练情况看是否需要加入CornerPooling等操作
        self.out_center = nn.Conv2d(inp_dim, 1, out_kernel, 1, 0 if out_kernel == 1 else 1)                             
        self.out_wh = nn.Conv2d(inp_dim, 2, out_kernel, 1, 0 if out_kernel == 1 else 1)
        self.out_offset = nn.Conv2d(inp_dim, 2, out_kernel, 1, 0 if out_kernel == 1 else 1)
        
        in_features = int(image_size[0] * image_size[1] / 16)  # 下采样率是4，所以除以16
        self.kpt_feature = BRC(inp_dim, oup_dim, 1, 1, 0)               
        self.pred_x = nn.Linear(in_features, int(image_size[0] * k)) 
        self.pred_y = nn.Linear(in_features, int(image_size[1] * k))
        
        # tag
        # self.tag_x = nn.Sequential(
        #     nn.Conv2d(oup_dim, oup_dim, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Flatten(start_dim=2),
        #     nn.Linear(in_features, int(image_size[0] * k)) 
        # )
        # self.tag_y = nn.Sequential(
        #     nn.Conv2d(oup_dim, oup_dim, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Flatten(start_dim=2),   # (b, c, h, w) -> (b,c, (h w))
        #     nn.Linear(in_features, int(image_size[0] * k)) 
        # )

    def forward(self, imgs):
        x = self.pre(imgs)

        for i in range(self.nstack):
            hg = self.hgs[i](x)

            feature = self.features[i](hg)
            x = self.merge[i](torch.cat([x, feature], dim=1))
         
        # predict bbox
        center = self.out_center(x)
        wh = self.out_wh(x)
        offset = self.out_offset(x)
        pred_centermap = torch.cat([center, wh, offset], dim=1)
        # predict keypoints
        x = self.kpt_feature(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        pred_x = self.pred_x(x)  # (b, c, w * k)
        pred_y = self.pred_y(x)  # (b, c, h * k)
        
        # tag_x = self.tag_x(x)
        # tag_y = self.tag_y(x)
        # return pred_centermap, pred_x, pred_y, tag_x, tag_y
        
        return pred_centermap, pred_x, pred_y
