HG_cfg = {
    "experiment_id": "litehrnet-region",
    'dataset': 'freihand',
    # model structure:
    'n_joints': 21,  # keypoints + region map
    'bn_momentum': 0.1,
    'need_region_map': True,  # 预测 region map
    'image_size': (256, 256), # (352, 352), 
    "hm_size": (64,),  # (16, 16, 32, 64), (22, 22, 44, 88),
    "hm_sigma": (2,), # (2, 2), 
    'mask_type': 3,
    "main_channels": 256,
    'increase': 0,
    'hg_depth': 4, 
    "nstack": 1,
    'higher_output': False,
    "param": (1, ),   # assert len(param) == nstack + higher_output
    

    # data augmentation
    'is_augment': True,
    'gamma_prob': 0.5,
    'sigmoid_prob': 0,
    'homography_prob': 0.5,
    'flip_prob': 0.5,

    # training setting
    'pin_memory': True,
    "loss_alpha": 2,
    "loss_beta": 0.25,
    "kpt_loss": "SmoothL1Loss",  # or 'L2Loss'
    "mask_loss": "L2Loss",
    "region_loss": "L2Loss",
    
    'batch_size': 32,
    "syncBN": True,         # 这个很关键,对最终性能有略微提升的效果
    'workers': 8,

    "CUDA_VISIBLE_DEVICES": "0, 1, 2, 3",
    'n_epochs': 60000,
    'eval_interval': 1,
    'pck_thr': 0.2, 
    'lr': 1e-3,
    'optim': 'AdamW',  # 'SGD', ' AdamW' or RMSprop
    "lr": 1e-3,
    'min_lr':1e-6,  # 周期正弦学习策略的最低学习率
    'T': 50,  # 周期
    'lr_gamma': 0.8,  # 每周期学习率递减因子
    "n_epochs": 500,
    "reload": True,
    "just_model": True,
    "checkpoint": "checkpoint/liteHRNet/lite-hrnet1/2021-12-12/81.598_mPCK_41epoch.pt",
    "save_root": "./checkpoint/liteHRNet/"
}