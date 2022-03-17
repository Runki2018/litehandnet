import torch
from torch import nn
import math
import os
from thop import profile

def get_SRhandNet(cfg):
    """
    :input: (batch, 3, h, w)
    :ouput: (tuple)(
        (btach, 24, 22, 22),
        (btach, 24, 22, 22),
        (btach, 24, 44, 44),
        (btach, 24, 88, 88)
    )
    """
    # SRHandNet PyTorch Model: https://www.yangangwang.com/papers/WANG-SRH-2019-07.html
    model_path = 'models/pose_estimation/srhandnet.pts'
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))

    # init model's parameters to train on FreiHand from scratch
    if not cfg['reload']:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0., std=0.05)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    return model

if __name__ == '__main__':
    net = get_SRhandNet()
    a = torch.rand(1, 3, 256, 256)
    out = net(a)
    print(f"{len(out)=}")
    for y in out:
        print(y.shape)
    # Pytorch 自带的计算参数量方法
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    
    # state = net.state_dict()
    # for k, v in state.items():
    #     print(f"{k}\t{v.shape=}")
    # print(net)
    # print('-' * 100)
    
    
    
    
    
    # 通过ptflops 计算
    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(net, (3,32,32), as_strings=True,print_per_layer_stat=True)
    
    # print("%s |flops: %s |params: %s" % (args.model_name,flops,params))
    
    # 通过 thop 计算
    # macs, params = profile(net, inputs=(a, ))
    # print(f"{macs=}\t{params=}")
    # from thop import clever_format
    # macs, params = clever_format([macs, params], "%.3f")
    # print(f"{macs=}\t{params=}")
    


