config_dict1 = {
    "experiment_id": "RKNet_mpii",
    "dataset": "zhhand20",
    "desc": "全新的网络",
    "backbone": "hourglass",

    # data argumentation
    "gamma_prob": 0.5,
    "sigmoid_prob": 0.5,
    'homography_prob': 0.5,
    'flip_prob': 0.5,

    # model
    "mask_type": 2,
    "hm_size": 128,
    "hm_sigma": (2, 3, 4, 5),
    "image_size": (512, 512),
    "n_joints": 21,
    "main_channels": 64,
    "num_hourglass": 2,

    # loss setting
    "loss_alpha": 2,
    "loss_beta": 0.25,
    "param": (1, 1.5, 2, 2, 2),
    "loss_flag": (True, True, True, True, True),
    "kpt_loss": "Focal",
    "mask_loss": "MaskLoss",
    "region_loss": "RegionLoss",

    # training setting
    "CUDA_VISIBLE_DEVICES": "0,",
    "device_id": 0,
    "is_augment": True,
    "batch_size": 1,
    "workers": 1,
    "step_size": 100,
    "lr": 1e-3,
    "n_epochs": 3000,
    "reload": False,
    "just_model": True,
    "checkpoint": "./weight/0.925_mPCK_handnet3_ms_512_zhhand.pt",
    # "checkpoint": "./weight/0.944_mPCK_handnet3_ms_512_freihand.pt",
    "save_root": "../record/"
}

current_config = config_dict1
