# w128 MSRA
cfg = dict(
    ID=1,
    MODEL=dict(
        name='litehrnet',
        depth=30,
        output_channel=16,  # num_joints + 3 region map
        pred_bbox=True,          # 模型是否预测边界框, 是则不进行旋转变换
        ),

    DATASET=dict(
        name='mpii_action',
        num_joints=16,
        image_size=[256, 256],
        heatmap_size=[64, 64],

        train=dict(ann_file='data/body/mpii/annotations_w_cat/train.json',
                   img_prefix='data/body/mpii/images/'),
        val=dict(ann_file='data/body/mpii/annotations_w_cat/val.json',
                 img_prefix='data/body/mpii/images/'),
        test=dict( ann_file='data/body/mpii/annotations_w_cat/val.json',
                  img_prefix='data/body/mpii/images/')
        ),

    PIPELINE=dict(
        # TopDownRandomFlip
        flip_prob=0.5,
        # TopDownGetRandomScaleRotation
        rot_prob=0,                     # 在目标检测中或带有region map 不进行旋转操作 0
        rot_factor=0,                   # 在目标检测或带有region map 不进行旋转操作 0
        scale_factor=0.3,
        # TopDownAffine
        use_udp=False,                   # 无偏数据处理, 与encoding='UDP'一起使用。
        # TopDownGenerateTarget
        sigma=2,
        kernel=(11, 11),                 # MSRA unbias编码时用到，sigma=2 => kernel=11
        encoding='MSRA',                 # MSRA | UDP
        unbiased_encoding=True,         # DARK中的编码方法，在MSRA生成热图时用到，整张热图都生成高斯值
        target_type='GaussianHeatmap',   # 用到不到，默认Gaussian就好
        simdr_split_ratio=0              # (int) simdr表征的放大倍率。 0, 1, 2, 3, 表示不使用Simdr
    ),

    CHECKPOINT=dict(interval=10, resume=False, load_best=True, save_root='checkpoints/'),
    EVAL=dict(interval=1,
              metric=['PCKh', 'AUC', 'EPE'],
              save_best='PCKh',
              pck_threshold=0.2),

    TRAIN=dict(
        distributed=True,
        pin_memory=False,
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        find_unused_parameters=False,
        workers=4,
        syncBN=False,
        total_epoches=210,
        batch_per_gpu=8,   # batch_size
    ),

    # 'Adam', 'SGD'
    OPTIMIZER=dict(type='Adam', lr=5e-4, warmup_steps=100),

    LOSS=dict(
        type='TopdownHeatmapLoss',
        loss_weight=[1.],   # 四个输出的权重
        auto_weight=False,
        with_simdr=False,  # (int => 0, 1, 2) 0表示不使用，1表示编码宽高为原图一倍大小
    )
)

def _get_cfg():
    return cfg