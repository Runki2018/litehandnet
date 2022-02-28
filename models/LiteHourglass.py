from threading import main_thread
from pyparsing import TokenConverter
from pyrsistent import inc
import torch
from torch import nn
from utils.training_kits import load_pretrained_state
from config.config import config_dict as cfg
from models.attention import RegionChannelAttention, SELayer, NAM_Channel_Att
from models.layers import DWConv, InvertedResidual, LiteHG, LiteResidual,  LiteBRC, SplitDWConv, BRC
from einops import rearrange, repeat


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.conv = nn.Sequential(
            BRC(inp_dim, out_dim // 2, 1, 1, 0),
            BRC(out_dim // 2, out_dim // 2, 3, 1, 1),
            BRC(out_dim // 2, out_dim, 1, 1, 0), 
        )

        if inp_dim == out_dim:
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = nn.Conv2d(inp_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        residual = self.skip_layer(x)
        out = self.conv(x)
        out += residual
        return out

class StemBlock(nn.Module):
    """ PELEE STEM Block的变体我在Conv1中再加了一个3x3卷积，来提高stem的初始感受野"""
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

class LiteHourglass(nn.Module):
    def __init__(self, nstack=1, inp_dim=64):
        super(LiteHourglass, self).__init__()
        oup_dim = cfg['n_joints'] + 3
        
        self.pre = StemBlock(out_channel=inp_dim)

        self.hgs = nn.ModuleList([LiteHG(4, inp_dim) for _ in range(nstack) ])
                   
        self.features = nn.ModuleList([
                            nn.Sequential(
                                # LiteResidual(inp_dim, inp_dim),
                                # LiteBRC(inp_dim, inp_dim),
                                Residual(inp_dim, inp_dim),
                                # BRC(inp_dim, inp_dim),
                                ) for _ in range(nstack)])
                                
        self.outs = nn.ModuleList([
            nn.Conv2d(inp_dim, oup_dim, 1, 1, 0) for _ in range(nstack)])

        self.merge_features = nn.ModuleList([
            nn.Conv2d(inp_dim, inp_dim, 1, 1, 0) for _ in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([
            nn.Conv2d(oup_dim, inp_dim, 1, 1, 0) for _ in range(nstack - 1)])
        
        self.nstack = nstack
        
        image_size = cfg['image_size']  # (w, h)
        k = cfg['simdr_split_ratio']  # default k = 2 
        in_features = int(image_size[0] * image_size[1] / (4 ** 2))  # 下采样率是4，所以除以16
        # self.vector_feature = Residual(inp_dim, cfg['n_joints'])  
        self.pred_x = nn.Linear(in_features, int(image_size[0] * k)) 
        self.pred_y = nn.Linear(in_features, int(image_size[1] * k))
        self.image_size = image_size

    def forward(self, imgs):
        x = self.pre(imgs)

        hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)

            feature = self.features[i](hg)
            hm_preds.append(self.outs[i](feature))
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](hm_preds[-1]) + self.merge_features[i](feature)
        
        # predict keypoints
        kpts = hm_preds[-1][:, 3:]
        # kpts = self.vector_feature(feature)
        kpts = rearrange(kpts, 'b c h w -> b c (h w)')
        pred_x = self.pred_x(kpts)  # (b, c, w * k)
        pred_y = self.pred_y(kpts)  # (b, c, h * k)   
        return hm_preds, pred_x, pred_y

