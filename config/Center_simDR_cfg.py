HG_cfg = {
    "experiment_id": "1HG-lite",
    'dataset': 'freihand100',
    # model structure:
    'n_joints': 21,  # keypoints + region map
    'gt_mode': {
        'region_map': False,
        'just_kpts': False,
        'center_map': True
        },  # 预测 region map
    'image_size': (256, 256), # (352, 352), 
    "hm_size": (64, 64),  # (16, 16, 32, 64), (22, 22, 44, 88),
    'hg_channels': [64, 128, 128, 128, 256], 
    "hm_sigma": (2, ), # (2, 2), 
    'out_kernel': 1, # 1 or 3
    'simdr_split_ratio' : 2,
    "nstack": 1,
    "param": (1, 1, 1),   # assert len(param) == nstack + higher_output
    
    # data augmentation
    'is_augment': True,
    'bbox_alpha': 1.5, # 关键点边框放大倍率
    'gamma_prob': 0.5,
    'sigmoid_prob': 0,
    'homography_prob': 0.5,
    'flip_prob': 0.5,
    'use_different_joints_weight': False,

    # training setting
    'pin_memory': True,
    "loss_alpha": 2,
    "loss_beta": 0.25,
    "kpt_loss": "SmoothL1Loss",  # or 'L2Loss'
    "mask_loss": "L2Loss",
    "region_loss": "L2Loss",
    
    'batch_size': 10,
    "syncBN": True,         # 这个很关键,对最终性能有略微提升的效果
    'workers': 10,

    "CUDA_VISIBLE_DEVICES": "0, 1, 2, 3",
    'n_epochs': 210,
    'eval_interval': 1,
    'pck_thr': 0.2, 
    'lr': 1e-3,
    'optim': 'AdamW',  # 'SGD', ' AdamW' or RMSprop
    "lr": 1e-4,
    'min_lr':1e-7,  # 周期正弦学习策略的最低学习率
    'T': 50,  # 周期
    'lr_gamma': 0.8,  # 每周期学习率递减因子
    "n_epochs": 500,
    "reload": False,
    "just_model": False,
    "checkpoint": "./checkpoint/Center_SimDR/1HG-lite/2022-01-06/0.269_PCK_42epoch.pt",
    "save_root": "./checkpoint/Center_SimDR/"
}