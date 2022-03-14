from re import S
import torch
from torch import nn
from utils.training_kits import load_pretrained_state
from config import config_dict as cfg
from models.attention import Flatten, SELayer, NAM_Channel_Att
from models.layers import DWConv, SplitDWConv, Residual
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
    
class ScaleAttConv(nn.Module):
    """
    https://blog.csdn.net/KevinZ5111/article/details/104730835?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-4-104730835.pc_agg_new_rank&utm_term=block%E6%94%B9%E8%BF%9B+residual&spm=1000.2123.3001.4430
    """
    def __init__(self, in_c, out_c, n_size=4):
        super().__init__()

        self.n_size = n_size
        mid_c = in_c 
        self.input_conv = BRC(in_c, mid_c, 1, 1, 0)

        scale_conv = []
        for i in range(1, n_size + 1):
            scale_conv.append(nn.ModuleList([ 
                DWConv(mid_c, mid_c, dilation=i, padding=i),
                DWConv(mid_c, mid_c, dilation=i, padding=i)
            ]))
        self.scale_conv = nn.ModuleList(scale_conv)
         
        self.mid_conv = nn.ModuleList([
            BRC(mid_c * n_size, mid_c, 1, 1, 0),
            BRC(mid_c * n_size, in_c, 1, 1, 0)
        ])
           
        self.output_conv = BRC(in_c, out_c, 1, 1, 0, bias=False)
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
        m = self.input_conv(x)   
        for i in range(2):
            scale_features = []
            for j in range(self.n_size):
                scale_features.append(self.scale_conv[j][i](m))
            m = torch.cat(scale_features, dim=1)
            m = self.mid_conv[i](m)

        features = m + x
        out = self.output_conv(features)
        b, c, _, _ = out.shape
        out = out * self.att(out).view(b, c, 1, 1)
        return out

class Hourglass_module(nn.Module):
    def __init__(self, n, channels, n_modules=[1, 1, 1, 1], n_sizes=4):
        super(Hourglass_module, self).__init__()

        assert n == len(channels), 'Error: num of layer != num of channels'
        self.n_stack = n
        
        self.skip_layer = nn.ModuleList([])
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        # TODO： 简化skip 和 最小层度的卷积层 是否会精度下降？    
        # 会，特别是skip层， 其中DWConvBlock效果比DWConv差很多
        for i in range(n):
            self.skip_layer.append(ScaleAttConv(channels[i], channels[i]))  
            if i != n-1:
                self.encoder.append(ScaleAttConv(channels[i], channels[i+1]))
                self.decoder.append(ScaleAttConv(channels[i+1], channels[i]))        
            else:
                self.encoder.append(ScaleAttConv(channels[i], channels[i], 2))
                self.decoder.append(ScaleAttConv(channels[i], channels[i], 2))  

        self.decoder = self.decoder[::-1]
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        # encoder
        out_skip = []
        for i in range(self.n_stack):
            out_skip.append(self.skip_layer[i](x))
            if i != self.n_stack -1:
                x = self.pool(x)
            x = self.encoder[i](x)
        # decoder
        # second_last_out = None
        for i in range(self.n_stack):
            x = self.decoder[i](x)
            x = x + out_skip.pop()
            # if i == self.n_stack - 2:
            #     second_last_out = x
            if i != self.n_stack -1:
                x = self.upsample(x)
        # return x, second_last_out
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
        self.channels = cfg['hg_channels']
        out_kernel = cfg['out_kernel']
        self.image_size = cfg['image_size']  # (w, h)
        k = cfg['simdr_split_ratio']  # default k = 2 
        assert out_kernel in [1, 3], 'Error: kernel size not in [1, 3]'
        inp_dim = self.channels[0]
        oup_dim = cfg['n_joints'] + 3

        self.nstack = nstack
        self.pre = my_pelee_stem(out_channel=inp_dim)

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass_module(len(self.channels), self.channels),
            ) for _ in range(self.nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                # Residual(inp_dim, inp_dim *2),
                ScaleAttConv(inp_dim, inp_dim *2 ),
                BRC(inp_dim * 2, inp_dim, 1, 1, 0),
            ) for _ in range(self.nstack)])
        
        # 去除了中间监督和简化了融合模块
        self.merge = nn.ModuleList([BRC(inp_dim * 2, inp_dim, 1, 1, 0) for _ in range(self.nstack - 1)]) 

        padding = 0 if out_kernel == 1 else 1
        self.outs = nn.ModuleList([nn.Conv2d(inp_dim, oup_dim, out_kernel, 1, padding) for _ in range(self.nstack)])
        
        # sindr at the second last layer of the hourglass module, 下采样率4
        in_features = int(self.image_size[0] * self.image_size[1] / (4 ** 2))
        self.pred_x = nn.Linear(in_features, int(self.image_size[0] * k)) 
        self.pred_y = nn.Linear(in_features, int(self.image_size[1] * k))
    
    def forward(self, imgs):
        # cycle_detection = False if imgs.shape[2] == self.image_size[1] else True
        
        x = self.pre(imgs)
        pred_hm, pred_x, pred_y = [], [], []
        for i in range(self.nstack):
            hg = self.hgs[i](x)

            feature = self.features[i](hg)
            pred_hm.append(self.outs[i](feature))
            
            if i < self.nstack - 1 :
                x = self.merge[i](torch.cat([x, feature], dim=1))

            kpts = pred_hm[-1][:, 3:]
            kpts = rearrange(kpts, 'b c h w -> b c (h w)')
            pred_x.append(self.pred_x(kpts))  # (b, c, w * k)
            pred_y.append(self.pred_y(kpts))  # (b, c, h * k)     
            
        
        # predict keypoints
        # if cycle_detection:
        #     kpts = pred_hm[-1][:, 3:]
        # else:
        #     kpts = self.second_features(second_last)
        # x = rearrange(kpts, 'b c h w -> b c (h w)')
        # pred_x = self.pred_x(x)  # (b, c, w * k)
        # pred_y = self.pred_y(x)  # (b, c, h * k)
            
        return pred_hm, pred_x, pred_y
        # return pred_hm
