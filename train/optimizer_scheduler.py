import imp
import torch
from torch import optim
import numpy as np

def get_optimizer(cfg, model, criterion):
    
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg['auto_weight']:  # 学习MTL的loss的权重
        params += [p for p in criterion.parameters() if p.requires_grad]
    
    assert cfg['optim'] in dir(optim), "optimizer type {} is not supported!!".format(cfg['optim'] )
    
    if cfg['optim'] == 'SGD':
            optimizer = optim.SGD(params, lr=cfg['lr'],
                                  momentum=0.9,
                                  weight_decay=0.0001,
                                  nesterov=False)
    else:
        optimizer = eval("optim." + cfg['optim'])(params, lr=cfg['lr'])

    return optimizer

  
def get_scheduler(cfg, optimizer, FP16_ENABLED=False, last_epoch=-1):
      # 自定义的逐周期递减正弦学习率曲线
    T, lr_gamma = cfg['T'], cfg['lr_gamma']
    lambda1 = lambda epoch: np.cos((epoch % (T + (epoch / T)) / (T + (epoch / T))) * np.pi / 2) * (lr_gamma ** (epoch // T))
    
    if FP16_ENABLED:
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer.optimizer,lambda1)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer.optimizer, [80, 100], 0.1)
    else:
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [80, 100], 0.1)
    
    scheduler.last_epoch = last_epoch
    return scheduler


