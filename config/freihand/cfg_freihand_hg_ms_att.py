cfg = {
    'dataset': 'freihand',
    'model': 'hg_ms_att',
    
    # model structure:
    'with_region_map': True,  # 是否使用region map
    'n_joints': 21,  # keypoints + region map
    'image_size': [256, 256], # (352, 352), 
    "hm_size": [64],  # (16, 16, 32, 64), (22, 22, 44, 88),
    "hm_sigma": [2], # (2, 2), 
    'hg_depth': 4,  # 一个沙漏模块的分辨率层数
    'out_kernel': 1, # 1 or 3
    'main_channels': 128,
    'simdr_split_ratio' : 2,
    'increase': 0,
 
    # data augmentation
    'is_augment': True,   # ! 在使用 freihand_plus 没作用
    'bbox_alpha': 1.1, # 关键点边框放大倍率, 预测框偏大
    'gamma_prob': 1,
    'sigmoid_prob': 1,
    'homography_prob': 1,
    'flip_prob': 1,
    'use_different_joints_weight': False,

    # training setting
    "distributed": True,
    "with_cycle_detection": True,
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    'cycle_detection_reduction': 1,  # ! 循环检测分辨率下降因子
    'find_unused_parameters': False,
    'pin_memory': True,
    'batch_size': 64,
    "syncBN": True,         # 这个很关键,对最终性能有略微提升的效果
    'workers': 4,
    'optim': 'SGD',  # 'SGD', ' AdamW' or RMSprop
    "lr": 1e-5,  # 初始学习率，最小学习率约为 0.01 * lr
    'T': 30,  # 周期
    'lr_gamma': 0.8,  # 每周期学习率递减因子
    "end_epoch": 150,
    'seed': 1,
    
    "reload": True,
    "checkpoint": 'output/freihand/steplr/freihand_plus_hg_ms_att/2022-03-20/best_pck.pt',
    "save_root": "./output/freihand/",
    
    # evaluation
    'eval_interval': 1,
    'pck_thr': 0.2,
    'DARK': True, 
    
    # loss
    "num_loss": 3,   # 损失函数的类别格式 一般有 hm_loss, region_loss
    "auto_weight": False,  # 是否使用多任务学习自动求权重
    
    "hm_loss_factor": [1.0],
    "region_loss_factor": [1.0],
    "kpt_loss": 'SmoothL1Loss',
    "region_loss":'SmoothL1Loss',
}

def _get_cfg():
    return cfg