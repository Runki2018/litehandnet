import torch
from torch import nn
from torch.nn.modules.utils import _triple, _pair, _single


# import softpool_cuda
#
# class CUDA_SOFTPOOL2d(Function):
#     @staticmethod
#     @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
#     def forward(ctx, input, kernel=2, stride=None):
#         # Create contiguous tensor (if tensor is not contiguous)
#         no_batch = False
#         if len(input.size()) == 3:
#             no_batch = True
#             input.unsqueeze_(0)
#         B, C, H, W = input.size()
#         kernel = _pair(kernel)
#         if stride is None:
#             stride = kernel
#         else:
#             stride = _pair(stride)
#         oH = (H - kernel[0]) // stride[0] + 1
#         oW = (W - kernel[1]) // stride[1] + 1
#         output = input.new_zeros((B, C, oH, oW))
#         softpool_cuda.forward_2d(input.contiguous(), kernel, stride, output)
#         ctx.save_for_backward(input)
#         ctx.kernel = kernel
#         ctx.stride = stride
#         if no_batch:
#             return output.squeeze_(0)
#         return output


class SoftPooling(nn.Module):
    """
    https://cloud.tencent.com/developer/article/1774834
    论文作者的实现：https://github.com/alexandrosstergiou/SoftPool/tree/master/pytorch/SoftPool
    别人的实现： https://blog.csdn.net/shanglianlm/article/details/115244769
    刚看到的博客，softPooling的性能和运算量都相比AcgPool和MaxPool差距不大，运算上也快，由于其保留图像细节的能力更好，所以
    想将它用到自己的沙漏模块和通道注意力模块中。
    """

    def __init__(self, kernel_size=2, stride=2):
        super(SoftPooling, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride)
        # todo 原作者的实验运行速度会更快。

    def forward(self, x):
        x_exp = torch.exp(x)
        # 借助两次均值池化来实现soft pool，但这样会降低性能
        x_exp_pool = self.avg_pool(x_exp)
        x = self.avg_pool(x_exp * x)
        return x / x_exp_pool


class StageChannelAttention(nn.Module):
    """
    ChannelAttention好，还是叫stageAttention, blockAttention好呢？
    https://blog.csdn.net/dedell/article/details/106768052
    参考SKNet和SENet，一个沙漏模块的预测热图，是从当前模块输出和前面所有沙漏模块输出中选出得分高的通道组合成最终输出。
    我们可以发现一个这样的问题，沙漏网络堆叠多个模块，其实是起到refine的作用，虽然模型最后一般来说后面的沙漏模块预测性能更好，
    但是在训练过程中，前面的沙漏模块会先收敛，那么在训练过程中其实前面的模块预测结果是更好的，那么能不能选出前面好的预测结果
    来作为此次的结果呢？ 所以加入通道注意力模块，从之前的预测结果中选出预测结果较好的热图通道。
    自己的贡献：
    1、使用通道注意力模块，让网络自己去从多个阶段中挑选最后的输出热图
    2、使用LayerNorm,去标准化同一图片，不同热图的全局值，好的预测热图，应该有着类似标准差和方差，
    3、怎么使得多个通道之间有联系
    如一张图片上有两个目标，则所有通道的热图，上应该都有两个峰值点，通过LayerNorm可以标准化所有通道的值，让模型去挑选更接近均值的通道。
    """

    def __init__(self, channel, reduction=4, n_block=2, min_unit=16):
        super(StageChannelAttention, self).__init__()
        self.n_block = n_block
        mid_channel = max(channel // reduction, min_unit)
        self.global_pool = nn.ModuleList([  # 这里看后面能不能改成全局 SoftPool
            nn.AdaptiveAvgPool2d(1) for _ in range(n_block)
        ])

        self.fc_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(channel),  # todo 验证自己加的 LayerNorm是否有作用
                nn.Linear(channel, mid_channel, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid_channel, channel),
                nn.Sigmoid(),
            ) for _ in range(n_block)
        ])
        # self.priority = torch.linspace(1, 2, n_block).view(-1, 1)  # Manually prioritize all output of blocks
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: [block_1, ..., block_n], block.shape=(batch, channel, h, w)
        batch, channel, _, _ = x[0].shape
        attention_vectors = torch.zeros((batch, self.n_block, channel), dtype=torch.float32).to(x[0].device)
        for i, block_i in enumerate(x):
            global_feature = self.global_pool[i](block_i).squeeze_()  # (batch, channel, 1, 1) -> (batch, channel)
            vector = self.fc_blocks[i](global_feature)  # (batch, channel)
            attention_vectors[:, i] = vector

        # attention_vectors = attention_vectors * self.priority.to(x[0].device)
        attention_vectors = self.softmax(attention_vectors)
        out = 0
        for i, block_i in enumerate(x):
            out += block_i * attention_vectors[:, i].view(batch, channel, 1, 1)
        out = out / self.n_block
        return out


class StageChannelAttention_all(nn.Module):
    """
    我希望能够融合多个块的特征来产生注意力，而不是仅仅一个通道自己的注意力
    ChannelAttention好，还是叫stageAttention, blockAttention好呢？
    https://blog.csdn.net/dedell/article/details/106768052
    参考SKNet和SENet，一个沙漏模块的预测热图，是从当前模块输出和前面所有沙漏模块输出中选出得分高的通道组合成最终输出。
    我们可以发现一个这样的问题，沙漏网络堆叠多个模块，其实是起到refine的作用，虽然模型最后一般来说后面的沙漏模块预测性能更好，
    但是在训练过程中，前面的沙漏模块会先收敛，那么在训练过程中其实前面的模块预测结果是更好的，那么能不能选出前面好的预测结果
    来作为此次的结果呢？ 所以加入通道注意力模块，从之前的预测结果中选出预测结果较好的热图通道。
    自己的贡献：
    1、使用通道注意力模块，让网络自己去从多个阶段中挑选最后的输出热图
    2、使用LayerNorm,去标准化同一图片，不同热图的全局值，好的预测热图，应该有着类似标准差和方差，
    3、怎么使得多个通道之间有联系
    如一张图片上有两个目标，则所有通道的热图，上应该都有两个峰值点，通过LayerNorm可以标准化所有通道的值，让模型去挑选更接近均值的通道。
    """

    def __init__(self, channel, reduction=4, n_block=2, min_unit=16):
        super(StageChannelAttention_all, self).__init__()
        self.n_block = n_block
        mid_channel = max(channel // reduction, min_unit)
        self.global_pool = nn.ModuleList([  # 这里看后面能不能改成全局 SoftPool
            nn.AdaptiveAvgPool2d(1) for _ in range(n_block)
        ])

        self.fc1_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(channel),  # todo 验证自己加的 LayerNorm是否有作用
                nn.Linear(channel, mid_channel, bias=False),
                nn.Dropout(p=0.3),
                nn.ReLU(),
            ) for _ in range(n_block)
        ])
        self.fc2_blocks = nn.ModuleList([
            nn.Linear(n_block * mid_channel, channel) for _ in range(n_block)
        ])
        # self.priority = torch.linspace(1, 2, n_block).view(-1, 1)  # Manually prioritize all output of blocks
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: [block_1, ..., block_n], block.shape=(batch, channel, h, w)
        batch, channel, _, _ = x[0].shape
        attention_vectors = torch.zeros((batch, self.n_block, channel), dtype=torch.float32).to(x[0].device)
        vectors = None  # (batch, n_block * mid_channel)
        for i, block_i in enumerate(x):
            # (batch, channel, 1, 1) -> (batch, channel)
            global_feature = self.global_pool[i](block_i).squeeze_(-1).squeeze_(-1)
            vector = self.fc1_blocks[i](global_feature)  # (batch, channel)
            if i == 0:
                vectors = vector
            else:
                vectors = torch.cat([vectors, vector], dim=1)

        # attention_vectors = attention_vectors * self.priority.to(x[0].device)
        for i in range(self.n_block):
            attention_vectors[:, i] = self.fc2_blocks[i](vectors)
        attention_vectors = self.softmax(attention_vectors)
        out = 0
        for i, block_i in enumerate(x):
            out += block_i * attention_vectors[:, i].view(batch, channel, 1, 1)

        # out = out / self.n_block
        return out


# --------------  Spatial and Channel Attention for Region maps : -----------------


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class RegionChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(RegionChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (batch, c, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # (batch, c, 1, 1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))  # (batch, c, 1, 1)
        max_out = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class RegionSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(RegionSpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, h, w)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)  # (batch, 2, h, w)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, reduction=16):
        super(CBAM, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
        )

        self.residual_conv = nn.Conv2d(inplanes, planes, 1, 1)
        self.ca = RegionChannelAttention(planes, reduction)
        self.sa = RegionSpatialAttention()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.pre(x)
        out = self.ca(out) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out += self.residual_conv(x)
        out = self.relu(out)
        return out


class SKConv(nn.Module):
    def __init__(self, channel, groups, reduction, n_scale=4, stride=1, min_unit=32):
        super(SKConv, self).__init__()
        d = max(int(channel / reduction), min_unit)

        self.features = channel
        self.convs = nn.ModuleList([  # 使用不同kernel size的卷积
            nn.Sequential(
                nn.Conv2d(channel,
                          channel,
                          kernel_size=(3 + i * 2, 3 + i * 2),
                          stride=(stride, stride),
                          padding=1 + i,
                          groups=groups), nn.BatchNorm2d(channel),
                nn.ReLU(inplace=False)
            ) for i in range(n_scale)
        ])

        self.fc = nn.Linear(channel, d)
        self.fcs = nn.ModuleList([nn.Linear(d, channel) for _ in range(n_scale)])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = None
        for i, conv in enumerate(self.convs):
            feature = conv(x).unsqueeze_(dim=1)
            features = feature if i == 0 else torch.cat([features, feature], dim=1)

        fea_U = torch.sum(features, dim=1)  # (batch, c, h, w) -> (batch, h, w)
        fea_s = fea_U.mean(-1).mean(-1)  #
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (features * attention_vectors).sum(dim=1)
        return fea_v


if __name__ == '__main__':
    a = torch.randn(2, 3, 6, 6)
    b = torch.randn(2, 3, 6, 6)
    # al = RegionChannelAttention(in_planes=3)
    # pred = al([a, b])
    sa = StageChannelAttention_all(3, 4, 2, 16)
    pred = sa([a, b])
    print(f"{pred.shape=}")
