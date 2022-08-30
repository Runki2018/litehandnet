import addict

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

    "bbox_factor": 1.3,  # 匹配关键点的限制区域为预测框的factor倍区域内
    "bbox_k": 3,   # 限制区域内每张热图取得分前k个候选点。
    "region_avg_kernel": 3,
    "region_avg_stride": 1,
    
    'blue_kernel': 19,  # 高斯模糊的核大小，DARK 默认为11
    'cd_iou': 0.3, # 循环检测阈值，边框IOU
    'cd_ratio': 0, # 循环检测阈值, 手部区域占比 
}


def get_config(cfg_path:str):
    assert isinstance(cfg_path, (str,)), "cfg_path should be a string"
    cfg_path = cfg_path.replace('.py','').replace('/', '.')
    print(f"load config => {cfg_path}")
    exec(f'from {cfg_path} import _get_cfg')
    cfg = eval('_get_cfg')()
    cfg = addict.Dict(cfg)

    # check correctness
    if cfg.MODEL.pred_bbox:
        cfg.PIPELINE.rot_prob = 0  # 预测边界框时，不进行旋转变换

    return cfg
