# from config.handNet6_cfg import current_config as config_dict
# from config.handNet_cfg import current_config as config_dict
# from config.RKNet_mpii_cfg import current_config as config_dict
# from config.RKNet_mpii_SA_cfg import current_config as config_dict
# from config.freihand_cfg.pose_estimation.srhandnet_cfg import srhandnet_cfg as config_dict
# from config.RKNet_handset_cfg import HG_cfg as config_dict
from config.Center_simDR_cfg import HG_cfg as config_dict

mpii = {
    "name": "mpii",
    "root": "./data/mpii/",
    "test_file": "./data/mpii/annot/valid.json",
    "train_file": "./data/mpii/annot/train.json",
    # "test_file": "../data/mpii/annot/train.json",
    "thr_list": [0.4091787741255851, 0.6927843836298436, 1.0551959497673846],
    "n_bbox": [8842, 7176, 5278, 950],
}


# ---  ZHhand 4104: ----
zhhand = {
    "name": "zhhand",
    "root": "/home/data2/ZHhands/allocate_dataset/",
    "test_file": "/home/data2/ZHhands/allocate_dataset/test.json",
    "train_file": "/home/data2/ZHhands/allocate_dataset/train.json",
    "thr_list": [0.1799627128063661, 0.33957693055802624, 0.6979341216965176],
    "n_bbox": [1771, 1610, 606, 117],
}

freihand = {
    "name": "freihand",
    "root": r"/root/data/Dataset/freiHAND/",
    "test_file": r"./split_testset/freihand/large.json",
    # "test_file": r"/root/data/Dataset/freiHAND/annotations/freihand_test.json",
    "train_file": r"/root/data/Dataset/freiHAND/annotations/freihand_train.json",
    "thr_list": [0.12740034726360222, 0.17060353893099983, 0.21433420109795465],
    "n_bbox": [19364, 34144, 35592, 15092],
}

Dataset = {
    "mpii": mpii,
    "zhhand": zhhand,
    "freihand": freihand,
}

seed = 1
DATASET = Dataset[config_dict["dataset"]]

# used in HeatmapParser.py
pcfg = {
    "num_candidates": 10,  # NMS前候选框个数，对应于取中心点热图前k个峰值点。
    "max_num_bbox": 1,    # 一张图片保留多少个预测框 == 最大预测目标数
    "nms_kernel": 11,        # 抑制掉热图峰值点区域范围
    "nms_stride": 1,
    "nms_padding": 5,
    "detection_threshold": 0.1,  # 框中心点是目标的阈值
    "iou_threshold": 0.6,  # NMS去掉重叠框的IOU阈值
    "tag_threshold": 1.,
    "use_detection_val": True,
    "ignore_too_much": True,

    "bbox_factor": 1.0,  # 匹配关键点的限制区域为预测框的factor倍区域内
    "bbox_k": 3,   # 限制区域内每张热图取得分前k个候选点。
    "region_avg_kernel": 3,
    "region_avg_stride": 1,
    
    'blue_kernel': 19,  # 高斯模糊的核大小，DARK 默认为11
    
    'cd_iou': 0.3, # 循环检测阈值，边框IOU
    'cd_ratio': 1, # 循环检测阈值, 手部区域占比 
}

