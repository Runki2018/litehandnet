srhandnet_cfg = {
    "experiment_id": "srhandnet256x256_freihand_1",
    'dataset': 'freihand',
    # model structure:
    'n_joints': 21,
    'bn_momentum': 0.1,
    'mutl_scale_hm': True,  # decoder 每层都有监督
    'need_region_map': True,  # 预测 region map
    'image_size':(256, 256), # (352, 352),
    "hm_size": (16, 16, 32, 64), # (22, 22, 44, 88),
    "hm_sigma": (2, 2, 2, 2), # (3, 3, 3, 3),
    'mask_type': 3,

    # data augmentation
    'is_augment': True,
    'gamma_prob': 0.5,
    'sigmoid_prob': 0,
    'homography_prob': 0.5,
    'flip_prob': 0.5,

    # training setting
    "loss_alpha": 2,
    "loss_beta": 0.25,
    "param": (1, 1, 1, 1),
    "kpt_loss": "L2Loss",
    "mask_loss": "L2Loss",
    "region_loss": "L2Loss",
    
    'batch_size': 64,
    "syncBN": True,
    'workers': 8,

    "CUDA_VISIBLE_DEVICES": "0, 1, 2, 3",
    'n_epochs': 60000,
    'eval_interval': 1,
    'pck_thr': 0.2, 
    'lr': 1e-3,
    'out_dir': './out_dir', 
    'optim': 'SGD',  # 'SGD', 'AdamW' or RMSprop
    "lr": 1e-3,
    'min_lr':1e-6,  # 周期正弦学习策略的最低学习率
    'T': 40,  # 周期
    'lr_gamma': 0.8,  # 每周期学习率递减因子
    "n_epochs": 500,
    "reload": True,
    "just_model": True,
    "checkpoint": "checkpoint/srhandnet256x256_freihand_1/2021-12-15/0.908_PCK_46epoch.pt",
    "save_root": "./checkpoint/"
}