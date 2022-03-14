import torch
from torch import nn
from utils.training_kits import load_pretrained_state
from models.pose_hg_ms_att import ME_att
from config import config_dict as cfg


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
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
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
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)

        low2 = self.low2(low1)

        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class HourglassNet(nn.Module):
    def __init__(self, nstack=cfg['nstack'], inp_dim=256, oup_dim=16,  increase=0,
        basic_block=Residual):
        super(HourglassNet, self).__init__()

        self.pre = nn.Sequential(
            # Conv(3, 64, 7, 2, bn=True, relu=True),
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Residual(64, 128),
            nn.MaxPool2d(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )

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
        self.load_weights()

    def forward(self, imgs):
        # our posenet
        # x = imgs.permute(0, 3, 1, 2)  # x of size 1,3,inpdim,inpdim
        # x = self.pre(x)
        x = self.pre(imgs)

        hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)

            feature = self.features[i](hg)
            preds = self.outs[i](feature)

            hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        return hm_preds
    
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
    net = Residual_SA(64, 64)
    a = torch.randn(2, 64, 5, 5)
    b = net(a)
    print(b.shape)
