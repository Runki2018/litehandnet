from threading import main_thread
import torch
from torch import nn
from utils.training_kits import load_pretrained_state
from config.config import config_dict as cfg
from models.attention import SELayer, NAM_Channel_Att
from models.layers import DWConv, DWConvBlock
from einops import rearrange, repeat


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

class MSRB(nn.Module):
    """
    https://blog.csdn.net/KevinZ5111/article/details/104730835?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-4-104730835.pc_agg_new_rank&utm_term=block%E6%94%B9%E8%BF%9B+residual&spm=1000.2123.3001.4430
    """
    def __init__(self, in_c, out_c):
        super(MSRB, self).__init__()

        mid_c = in_c // 4
        self.conv1 = BRC(in_c, mid_c, 1, 1, 0, bias=False)

        self.mid1_conv = nn.ModuleList([
            BRC(mid_c , mid_c, 3, 1, 1),
            BRC(2 * mid_c , 2 * mid_c, 3, 1, 1)])
        
        self.mid2_conv = nn.ModuleList([  # MSRB中是5x5，这里用空洞卷积来扩大感受野，可减少参数量
            BRC(mid_c, mid_c, 5, 1, 2),
            BRC(2 * mid_c, 2 * mid_c, 5, 1, 2)])
        
        self.conv2 = BRC(4 * mid_c, in_c, 1, 1, 0, bias=False)
        self.conv3 = BRC(in_c, out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        m = self.conv1(x)
        for i in range(2):
            m1 = self.mid1_conv[i](m)
            m2 = self.mid2_conv[i](m)
            m = torch.cat([m1, m2], dim=1)

        out = self.conv2(m) + x
        out = self.conv3(out)
        return out

class MSRB_T(nn.Module):
    """
    https://blog.csdn.net/KevinZ5111/article/details/104730835?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-4-104730835.pc_agg_new_rank&utm_term=block%E6%94%B9%E8%BF%9B+residual&spm=1000.2123.3001.4430
    """
    def __init__(self, in_c, out_c):
        super(MSRB_T, self).__init__()

        mid_c = in_c // 4
        self.conv1 = BRC(in_c, mid_c, 1, 1, 0, bias=False)

        self.mid1_conv = nn.ModuleList([
            BRC(mid_c , mid_c, 3, 1, 1),
            BRC(2 * mid_c , 2 * mid_c, 3, 1, 1)])
        
        self.mid2_conv = nn.ModuleList([ 
            nn.Sequential(
                BRC(mid_c, mid_c, 3, 1, 1),
                nn.Conv2d(mid_c, mid_c, 3, 1, 1)),
            nn.Sequential(
                BRC(2 * mid_c , 2 * mid_c, 3, 1, 1),
                nn.Conv2d(2 * mid_c , 2 * mid_c, 3, 1, 1))
            ])
            
        self.conv2 = BRC(4 * mid_c, in_c, 1, 1, 0, bias=False)
        self.conv3 = BRC(in_c, out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        m = self.conv1(x)
        for i in range(2):
            m1 = self.mid1_conv[i](m)
            m2 = self.mid2_conv[i](m)
            m = torch.cat([m1, m2], dim=1)

        out = self.conv2(m) + x
        out = self.conv3(out)
        return out

class MSRB_SE(nn.Module):
    """
    https://blog.csdn.net/KevinZ5111/article/details/104730835?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-4-104730835.pc_agg_new_rank&utm_term=block%E6%94%B9%E8%BF%9B+residual&spm=1000.2123.3001.4430
    """
    def __init__(self, in_c, out_c):
        super(MSRB_SE, self).__init__()
        self.global_pool = nn.Sequential(  # 这里看后面能不能改成全局 SoftPool
            nn.Dropout2d(p=0.3),
            nn.AdaptiveAvgPool2d(1))

        self.fc = nn.Sequential(
            nn.LayerNorm(in_c),
            nn.ELU(inplace=True),
            nn.Linear(in_c, in_c // 2),
            nn.BatchNorm1d(in_c // 2),
            nn.Linear(in_c // 2, in_c),   
            nn.Sigmoid())

        mid_c = in_c // 2
        self.conv1 = BRC(in_c, mid_c, 1, 1, 0)

        self.mid1_conv = nn.ModuleList([
            BRC(mid_c, mid_c // 2, 3, 1, 1)  for _ in range(2) ])
        
        self.mid2_conv = nn.ModuleList([  # MSRB中是5x5，这里用空洞卷积来扩大感受野，可减少参数量
            BRC(mid_c, mid_c // 2, 3, 1, 2, dilation=2)  for _ in range(2)])
        
        self.conv2 = BRC(mid_c, in_c, 1, 1, 0, bias=False)
        self.conv3 = BRC(in_c, out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        # (batch, channel, 1, 1) -> (batch, channel)
        b, c, _, _ = x.shape
        factors = self.global_pool(x).flatten(1)
        factors = self.fc(factors).view(b, c, 1, 1)

        m = self.conv1(x)
        for i in range(2):
            m1 = self.mid1_conv[i](m)
            m2 = self.mid2_conv[i](m)
            m = torch.cat([m1, m2], dim=1)

        features = self.conv2(m) * factors
        out = self.conv3(features)
        return out


class MSRB_D(nn.Module):
    """
    https://blog.csdn.net/KevinZ5111/article/details/104730835?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-4-104730835.pc_agg_new_rank&utm_term=block%E6%94%B9%E8%BF%9B+residual&spm=1000.2123.3001.4430
    """
    def __init__(self, in_c, out_c):
        super(MSRB_D, self).__init__()


        mid_c = in_c // 2
        self.conv1 = BRC(in_c, mid_c, 1, 1, 0)

        self.mid1_conv = nn.ModuleList([
            BRC(mid_c, mid_c // 2, 3, 1, 1)  for _ in range(2) ])
        
        self.mid2_conv = nn.ModuleList([  # MSRB中是5x5，这里用空洞卷积来扩大感受野，可减少参数量
            BRC(mid_c, mid_c // 2, 3, 1, 2, dilation=2)  for _ in range(2)])
        
        self.conv2 = BRC(mid_c, in_c, 1, 1, 0, bias=False)
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

class Hourglass_module(nn.Module):
    def __init__(self, n, f, increase=0, basic_block=Residual):
        super(Hourglass_module, self).__init__()
        nf = f + increase
        self.up1 = basic_block(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = basic_block(f, nf)
        # Recursive hourglass
        if n > 1:
            self.low2 = Hourglass_module(n - 1, nf)
        else:
            self.low2 = basic_block(nf, nf)
            # self.low2 = Residual(nf, nf)
        self.low3 = basic_block(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)

        low2 = self.low2(low1)

        low3 = self.low3(low2)
        up2 = self.up2(low3)
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

class ME_att_lite(nn.Module):
    """
    https://blog.csdn.net/KevinZ5111/article/details/104730835?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-4-104730835.pc_agg_new_rank&utm_term=block%E6%94%B9%E8%BF%9B+residual&spm=1000.2123.3001.4430
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        mid_c = in_c // 2
        self.conv1 = BRC(in_c, mid_c, 1, 1, 0)

        self.mid1_conv = nn.Sequential(
                DWConv(mid_c, mid_c // 2),
                DWConv(mid_c // 2, mid_c // 2)
            )

        self.mid2_conv =  nn.Sequential(
                DWConv(mid_c, mid_c // 2, dilation=2, padding=2),
                DWConv(mid_c // 2, mid_c // 2),
                )
  
        self.conv3 = DWConvBlock(mid_c * 2, out_c)
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
        m1 = self.mid1_conv(m)
        m2 = self.mid2_conv(m)
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
        
        self.conv2 = nn.Sequential(   
            nn.Conv2d(mid_channel * 2, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.conv1(x)
        b1 = self.branch1(out)
        b2 = self.branch2(out)
        out = torch.cat([b1, b2], dim=1)
        out = self.conv2(out)
        return out

class higher_output_layer(nn.Module):
    """学习HigherHRNet, 使用转置卷积放大最终输出"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.zoom = nn.ConvTranspose2d(in_c, in_c, kernel_size=4, stride=2, 
                                padding=1, output_padding=0, dilation=1)

        self.conv2 = nn.Sequential(
            MSRB_D_DWConv(in_c, in_c),
            DWConv(in_c, in_c),
            nn.Conv2d(in_c, out_c, 1, 1, 0)
        )
    def forward(self, x):
        out_zoom = self.zoom(x)
        out = self.conv2(out_zoom)
        return out


class higher_output_MS_layer(nn.Module):
    """学习HigherHRNet, 但并行了多种上采样方式各上采样一部分通道，但收敛速度太慢了"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, in_c // 2, 1, 1, 0)
        self.zoom = nn.ModuleList([
            nn.ConvTranspose2d(in_c // 2, in_c // 2, kernel_size=4, stride=2, 
                                padding=1, output_padding=0, dilation=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.UpsamplingBilinear2d(scale_factor=2)
        ])
         # mid_c = (c_convtranspose + c_upsample) + c_pixelshuffle 
        mid_c = int(in_c // 2 * 2 + in_c // 2 / 4) 
        self.conv2 = nn.Sequential(
            BRC(mid_c, in_c, 1, 1, 0),
            DWConv(in_c, in_c),
            DWConv(in_c, in_c),
            nn.Conv2d(in_c, out_c, 1, 1, 0)
        )
    def forward(self, x):
        out = self.conv1(x)
        out_zoom = [layer(out) for layer in self.zoom]
        out = torch.cat(out_zoom, dim=1)
        out = self.conv2(out)
        return out


class HourglassNet_SA(nn.Module):
    def __init__(self, nstack=cfg['nstack'], increase=cfg['increase'],
        basic_block=ME_att_lite):
        super(HourglassNet_SA, self).__init__()
        inp_dim = cfg['main_channels']
        oup_dim = cfg['n_joints'] + 3
        
        self.pre = my_pelee_stem(out_channel=inp_dim)

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass_module(cfg['hg_depth'], inp_dim, increase, basic_block=basic_block),
            ) for _ in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for _ in range(nstack)])

        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 3, relu=False, bn=False) for _ in range(nstack)])

        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for _ in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for _ in range(nstack - 1)])
        self.nstack = nstack
        # self.load_weights()
        # for p in self.parameters():  # froze weight
        #     p.requires_grad = False
        
        image_size = cfg['image_size']  # (w, h)
        k = cfg['simdr_split_ratio']  # default k = 2 
        in_features = int(image_size[0] * image_size[1] / (8 ** 2))  # 下采样率是4，所以除以16
        # self.vector_feature = Residual(inp_dim, cfg['n_joints'])  
        self.pred_x = nn.Linear(in_features, int(image_size[0] * k)) 
        self.pred_y = nn.Linear(in_features, int(image_size[1] * k))
        self.image_size = image_size

    def forward(self, imgs):
        x = self.pre(imgs)

        # hm_preds, pred_x , pred_y = [], [], []
        hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)

            feature = self.features[i](hg)
            hm_preds.append(self.outs[i](feature))
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](hm_preds[-1]) + self.merge_features[i](feature)
        
        # predict keypoints
        if imgs.shape[-1] != self.image_size[0]:
            kpts = hm_preds[-1][:, 3:]
            # kpts = self.vector_feature(feature)
            kpts = rearrange(kpts, 'b c h w -> b c (h w)')
            pred_x = self.pred_x(kpts)  # (b, c, w * k)
            pred_y = self.pred_y(kpts)  # (b, c, h * k)   
        else:
            pred_x, pred_y = None, None
        return hm_preds, pred_x, pred_y

    def load_weights(self):
        if self.nstack == 2:
            # pre_trained = './weight/2HG_checkpoint.pt'
            pre_trained = './weight/2HG_86.724PCK_76epoch.pt'
        elif self.nstack == 8:
            # pre_trained = './weight/8HG_checkpoint.pt'
            pre_trained = './weight/8HG_89.355PCK_90epoch.pt'
        else:
            return
        # pretrained_state = torch.load(pre_trained)['state_dict']
        pretrained_state = torch.load(pre_trained, map_location=torch.device('cpu'))['model_state']
        
        state, _ = load_pretrained_state(self.state_dict(),
                                                pretrained_state)

        self.load_state_dict(state, strict=False)
        # for p in self.parameters():
        #     p.requires_grad = False


if __name__ == '__main__':
    net = MSRB(64, 64)
    a = torch.randn(2, 64, 5, 5)
    b = net(a)
    print(b.shape)
