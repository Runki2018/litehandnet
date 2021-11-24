import torch
from torch import nn
from models.attention import SELayer, CBAM
from models.layers import Residual, Conv, BasicBlock, Bottleneck


class SizeBlock(nn.Module):
    """
    input: (B, C, H, W)
    output: (B, len(n_part) + 1, H, W)
    """

    def __init__(self, in_channel, width=64, height=64, n_part=(1, 2, 3, 4)):
        super(SizeBlock, self).__init__()

        self.multi_size = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, 1, 2 * i + 1, 1, i),
                nn.AdaptiveAvgPool2d((n, n)),
            ) for i, n in enumerate(n_part)
        ])
        self.UpSample = nn.Upsample(size=(height, width), mode='nearest')
        self.conv = nn.ModuleList([
            nn.Conv2d(1, 1, 1, 1) for _ in range(len(n_part) + 1)
        ])
        self.n_part = n_part
        # self.side_type = width if side_type == 'width' else height
        self.width = width
        self.height = height

    def forward(self, x):
        B, C, H, W = x.shape
        assert self.width == W and self.height == H, 'Size mismatch!'
        # out_list = [self.conv[-1](x)]
        out_list = []
        for i, n in enumerate(self.n_part):
            # ratio = 1 / n
            # out = self.UpSample(self.multi_size[i](x)) * ratio  # (B, C, n, n) -> (B, C, H, W)
            out = self.UpSample(self.multi_size[i](x))  # (B, C, n, n) -> (B, C, H, W)
            out = self.conv[i](out)
            out_list.append(out)

        out = torch.cat(out_list, dim=1)
        m, _ = torch.max(out, dim=1, keepdim=True)
        out = torch.cat([out, m], dim=1)
        return out


class RegionBlock(nn.Module):
    def __init__(self, in_dim, size=(64, 64), n_part=(1, 2, 3, 4), init_values=(0.5, 0.5)):
        super(RegionBlock, self).__init__()
        mid_dim1 = 2
        mid_dim2 = len(n_part) + 1
        self.branches = nn.ModuleList([
            nn.Sequential(
                Residual(in_dim, mid_dim1),
                SizeBlock(mid_dim1, size[0], size[1], n_part),
                Conv(mid_dim2, 1, 1, 1, bn=True, relu=False),
                nn.Sigmoid(),
            ) for _ in range(2)
        ])
        self.scale_factors = nn.ModuleList([
            nn.Sequential(
                Residual(in_dim, mid_dim1),
                Conv(mid_dim1, 1, 1, 1, bn=True, relu=False),
                nn.Sigmoid(),
            ) for _ in range(2)
        ])
        self.init_values = init_values

    def forward(self, x):
        two_branches = []
        # predict width and height heatmaps
        for i in range(2):
            out = self.branches[i](x)
            scale_factor = self.scale_factors[i](x) + self.init_values[i]
            # print(f"{scale_factor=}")
            out = out * scale_factor
            two_branches.append(out)
        out = torch.cat(two_branches, dim=1)
        return out


if __name__ == '__main__':
    p = RegionBlock(6, (64, 64), (1, 3, 2, 4))
    a = torch.randn(2, 6, 64, 64)
    a.requires_grad = True
    # output = block(a)
    output = p(a)
    print(f"{output.shape=}")
    print(f"{output.requires_grad=}")
    print(f"{output=}")

    # n = 2
    # H, W = 9, 9
    # h, w = H // n, W // n
    # for i in range(n):
    #     for j in range(n):
    #         print(f"[{i * h}:{(i+1)*h}, {j * w}:{(j+1)*w}]")
