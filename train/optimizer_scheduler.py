from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, ReduceLROnPlateau
import numpy as np
from train import adai_optim


def get_optimizer(cfg, model, criterion):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.LOSS.auto_weight or cfg.PIPELINE.simdr_split_ratio > 0:  # 学习MTL的loss的权重
        params += [p for p in criterion.parameters() if p.requires_grad]
    
    valid_optim = dir(optim) + ['Adai', 'AdaiW', 'adai', 'adaiw']
    assert cfg.OPTIMIZER.type in valid_optim, \
        "optimizer type {} is not supported!!".format(cfg.OPTIMIZER.type)
    
    if cfg.OPTIMIZER.type == 'SGD':
        optimizer = optim.SGD(params, lr=cfg.OPTIMIZER.lr, momentum=0.9,
                              weight_decay=1e-8, nesterov=False)
    elif cfg.OPTIMIZER.type.lower() == 'adai':
        optimizer = adai_optim.Adai(params, lr=cfg.OPTIMIZER.lr, betas=(0.1, 0.99),
                                    eps=1e-03, weight_decay=1e-8)
    elif cfg.OPTIMIZER.type.lower() == 'adaiw':
        optimizer = adai_optim.Adai(params, lr=cfg.OPTIMIZER.lr, betas=(0.1, 0.99),
                                    eps=1e-03, weight_decay=1e-8)
    else:
        optimizer = eval("optim." + cfg.OPTIMIZER.type)(params, lr=cfg.OPTIMIZER.lr)
    return optimizer


def get_scheduler(cfg, optimizer, FP16_ENABLED=False, last_epoch=-1):
    _optimizer = optimizer.optimizer if FP16_ENABLED else optimizer
    if cfg.OPTIMIZER.type in ['SGD', 'Adai', 'AdaiW', 'adai', 'adaiw']:
        # https://zhuanlan.zhihu.com/p/503146643
        scheduler = CosineAnnealingWarmRestarts(_optimizer, 10, 2)
        print(f"Use CosineAnnealingWarmRestarts scheduler")
        # scheduler = ReduceLROnPlateau(_optimizer, mode='min', factor=0.5, patience=5, verbose=True, cooldown=5, min_lr=cfg.OPTIMIZER.lr*1e-4)
        # print(f"Use ReduceLROnPlateau scheduler")
    else:
        step_epoch = cfg.OPTIMIZER.get('step_epoch', [170, 200])
        scheduler = MultiStepLR(_optimizer, step_epoch, 0.1)
        print(f"Use StepLR scheduler")

    scheduler.last_epoch = last_epoch
    return scheduler
