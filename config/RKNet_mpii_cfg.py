config_dict1 = {
    "experiment_id": "2HG_R1",
    "dataset": "mpii",
    "num_sample": -1,
    "backbone": "hourglass",
    "is_augment": True,

    # data setting
    'data_format': 'jpg',
    'scale_factor': 0.25,
    'rotation_factor': 30,
    'flip': False,
    'num_joints_half_body': 8,
    'prob_half_body': 0.0,
    'color_rgb': False,
    'select_data': False,  # 用于去掉一部分无用的样本
    'test_set': 'valid',  # 'valid' or 'test' used in evaluate()
    'pin_memory': True,
    'train_shuffle': True,
    'mask_type': 3,

    # model
    'target_type': 'gaussian',
    'image_size': (256, 256),
    "hm_size": (64, 64),
    "hm_sigma": (2,),  # todo 要不要也搞一个动态sigma
    'use_different_joints_wight': False,
    "n_joints": 16,
    "main_channels": 64,
    'increase': 32,
    "nstack": 2,
    'hg_depth': 5, 
    "n_part": (1, 2, 3, 4, 5),

    # loss setting
    "loss_alpha": 2,
    "loss_beta": 0.25,
    "param": (1, 1),
    "kpt_loss": "L2Loss",
    "mask_loss": "MaskLoss",
    "region_loss": "RegionLoss",

    # training setting
    "CUDA_VISIBLE_DEVICES": "0, 1, 2, 3",
    "batch_size": 28,  # batch_size = batch_per_gpu * num_gpus
    "syncBN": True,
    "workers": 14,
    'eval_interval': 1,
    'out_dir': './out_dir', 
    "lr": 1e-3,
    'min_lr':1e-6,  # 周期正弦学习策略的最低学习率
    'T': 15,  # 周期
    'lr_gamma': 0.9,  # 每周期学习率递减因子
    "n_epochs": 500,
    "reload": False,
    "just_model": False,
    "checkpoint": "./record/3HG_R1/2021-11-28/51.527_mPCK_7epoch.pt",
    "save_root": "./record/"
}

current_config = config_dict1
