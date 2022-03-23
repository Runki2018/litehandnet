cfg = {
    'dataset': 'freihand_plus',
    'model': 'litehrnet',
    
    # model structure:
    'with_region_map': True,
    'n_joints': 21,  # keypoints + region map
    'image_size': [256, 256], # (352, 352), 
    "hm_size": [64],  # (16, 16, 32, 64), (22, 22, 44, 88),
    "hm_sigma": [2], # (2, 2), 
    'simdr_split_ratio' : 0,  # ! 0表示不使用 simdr
 
    # data augmentation
    'is_augment': False,
    'bbox_alpha': 1.1, # 关键点边框放大倍率, 预测框偏大
    'gamma_prob': 0.2,
    'sigmoid_prob': 0.1,
    'homography_prob': 0.3,
    'flip_prob': 0.3,
    'use_different_joints_weight': False,  # todo 有时间可以验证其作用
    
    # training setting
    "distributed": True,
    'CUDA_VISIBLE_DEVICES': '0',
    "with_cycle_detection": False,
    'cycle_detection_reduction': 1,  # ! 循环检测分辨率下降因子
    'find_unused_parameters': False,
    'pin_memory': True,
    'batch_size': 32,
    "syncBN": True,         # 这个很关键,对最终性能有略微提升的效果
    'workers': 4,
    'optim': 'AdamW',  # 'SGD'| ' AdamW' | RMSprop | Adam
    "lr": 1e-3,  # 初始学习率，最小学习率约为 0.01 * lr
    'T': 50,  # 周期
    'lr_gamma': 0.9,  # 每周期学习率递减因子
    "end_epoch": 250,
    'seed': 1,
     
    "reload": True,
    "checkpoint": "output/freihand/freihand_plus_litehrnet/2022-03-21/best_pck.pt",
    "save_root": "./output/freihand/",
    
    # evaluation
    'eval_interval': 1,
    'pck_thr': 0.2, 
    'DARK': True, # 使用DARK 或 关键点向次峰值点偏移
    
    # loss
    "num_loss": 2,   # 损失函数的类别格式 一般有 hm_loss, region_loss
    "auto_weight": False,  # 是否使用多任务学习自动求权重
    
    "hm_loss_factor": [1.0],
    "region_loss_factor": [1.0],
    "kpt_loss": 'SmoothL1Loss',
    "region_loss":'SmoothL1Loss',
}

def _get_cfg():
    return cfg