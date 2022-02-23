HG_cfg = {
    "experiment_id": "1HG-original-c128-h4-k2-o64",
    'dataset': 'freihand',
    # model structure:
    'n_joints': 21,  # keypoints + region map
    'gt_mode': {
        'region_map': False,
        'just_kpts': False,
        'center_map': True
        },  # 预测 region map
    'image_size': (256, 256), # (352, 352), 
    "hm_size": (64, 64),  # (16, 16, 32, 64), (22, 22, 44, 88),
    "hm_sigma": (2, ), # (2, 2), 
    'hg_channels': [128, 128, 128, 128], 
    'hg_depth': 4,  # 一个沙漏模块的分辨率层数
    'higher_output': False,
    'out_kernel': 1, # 1 or 3
    'main_channels': 128,
    'simdr_split_ratio' : 2,
    'cycle_detection_reduction': 2,  # 循环检测分辨率下降因子
    "nstack": 1,  
    "param": (1, 1, 1),   # assert len(param) == nstack + higher_output
    'increase': 0,
 
    # data augmentation
    'is_augment': False,
    'bbox_alpha': 1.5, # 关键点边框放大倍率, 预测框偏大
    'gamma_prob': 0.2,
    'sigmoid_prob': 0.1,
    'homography_prob': 0.3,
    'flip_prob': 0.3,
    'use_different_joints_weight': False,

    # training setting
    'pin_memory': True,
    "loss_alpha": 2,
    "loss_beta": 0.25,
    "kpt_loss": "SmoothL1Loss",  # "SmoothL1Loss" or 'L2Loss'
    "mask_loss": "L2Loss",
    "region_loss": "SmoothL1Loss",

    'batch_size': 48,
    "syncBN": True,         # 这个很关键,对最终性能有略微提升的效果
    'workers': 8,

    "CUDA_VISIBLE_DEVICES": "0, 1, 2, 3",
    'eval_interval': 1,
    'pck_thr': 0.2, 
    'optim': 'AdamW',  # 'SGD', ' AdamW' or RMSprop
    "lr": 1e-3,  # 初始学习率，最小学习率约为 0.01 * lr
    'T': 50,  # 周期
    'lr_gamma': 0.8,  # 每周期学习率递减因子
    "n_epochs": 500,
    "reload": True,
    "just_model": True,
    "checkpoint": "./checkpoint/final_all/ME-att/1HG-original-c128-h4-k2-o64/2022-02-23/74.663_AP_0epoch.pt",
    "save_root": "./checkpoint/final_all/ME-att/"
}