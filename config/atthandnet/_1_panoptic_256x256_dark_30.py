# w128 best_encoding + SimDR
cfg = dict(
    ID=1,
    MODEL=dict(
        name='atthandnet',
        pred_bbox=True,          # 模型是否预测边界框, 是则不进行旋转变换
        ),

    DATASET=dict(
        name='panoptic',    # datasets.datasets.__init__
        num_joints=21,
        image_size=[256, 256],
        heatmap_size=[64, 64],

        train=dict(
            ann_file='data/handset/panoptic/annotations/panoptic_train.json',
            img_prefix='data/handset/panoptic/'),
        val=dict(
            ann_file='data/handset/panoptic/annotations/panoptic_test.json',
            img_prefix='data/handset/panoptic/'),
        test=dict(
            ann_file='data/handset/panoptic/annotations/panoptic_test.json',
            img_prefix='data/handset/panoptic/')
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
        unbiased_encoding=True,          # DARK中的编码方法，在MSRA生成热图时用到，整张热图都生成高斯值
        target_type='GaussianHeatmap',   # 用到不到，默认Gaussian就好
        # !(int) simdr表征的放大倍率。 0, 1, 2, 3, 表示不使用Simdr
        simdr_split_ratio=0,             
    ),

    CHECKPOINT=dict(interval=10, resume=True,
                    load_best=True, save_root='checkpoints/'),

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
        syncBN=True,
        total_epoches=210,
        batch_per_gpu=8,   # batch_size
    ),

    # 'Adam', 'SGD'
    OPTIMIZER=dict(type='Adam', lr=5e-4, warmup_steps=100, step_epoch=[170, 200]),

    LOSS=dict(
        type='TopdownHeatmapLoss',
        loss_weight=[1., 1.],   # 四个输出的权重
        auto_weight=False,
    )
)

def _get_cfg():
    return cfg