# w128 MSRA_DARK
cfg = dict(
    ID=4,
    MODEL=dict(
        name='srhandnet',
        output_channel=21,  # num_joints + 3 region map
        pred_bbox=False,     # 模型是否预测边界框, 是则不进行旋转变换, 以及不生成region map
        ),

    DATASET=dict(
        name='onehand10k',
        num_joints=21,
        image_size=[256, 256],
        heatmap_size=[[16, 16], [16, 16], [32, 32], [64, 64]],

        train=dict(
            ann_file='data/handset/OneHand10K/annotations/onehand10k_train.json',
            img_prefix='data/handset/OneHand10K/'),
        val=dict(
            ann_file='data/handset/OneHand10K/annotations/onehand10k_test.json',
            img_prefix='data/handset/OneHand10K/'),
        test=dict(
            ann_file='data/handset/OneHand10K/annotations/onehand10k_test.json',
            # ann_file='data/handset/OneHand10K/annotations/onehand10k_train.json',
            img_prefix='data/handset/OneHand10K/')
        ),

    PIPELINE=dict(
        # TopDownRandomFlip
        flip_prob=0.5,
        # TopDownGetRandomScaleRotation
        rot_prob=0.5,                     # 在目标检测中或带有region map 不进行旋转操作 0
        rot_factor=45,                   # 在目标检测或带有region map 不进行旋转操作 0
        scale_factor=0.3,
        # TopDownAffine
        use_udp=False,                   # 无偏数据处理, 与encoding='UDP'一起使用。
        # TopDownGenerateTarget
        sigma=[2, 2, 2, 2],
        kernel=(11, 11),                 # MSRA unbias编码时用到，sigma=2 => kernel=11
        encoding='MSRA',                 # MSRA | UDP
        unbiased_encoding=True,         # !DARK中的编码方法，在MSRA生成热图时用到，整张热图都生成高斯值
        target_type='GaussianHeatmap',   # 用到不到，默认Gaussian就好
        simdr_split_ratio=0              # (int) simdr表征的放大倍率。 0, 1, 2, 3, 表示不使用Simdr
    ),

    CHECKPOINT=dict(interval=10, resume=True, load_best=False, save_root='checkpoints/'),
    EVAL=dict(interval=1,
              metric=['PCK', 'AUC', 'EPE'],
              save_best='PCK',
              pck_threshold=0.2),

    TRAIN=dict(
        distributed=True,
        pin_memory=True,
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        find_unused_parameters=False,
        workers=2,
        syncBN=False,
        total_epoches=210,
        batch_per_gpu=16,   # ! batch_size
    ),

    # 'Adam', 'SGD'
    # OPTIMIZER=dict(type='Adam', lr=1e-3, warmup_steps=100, step_epoch=[170, 200]),
    # OPTIMIZER=dict(type='AdamW', lr=1e-3, warmup_steps=100, step_epoch=[170, 200]),
    OPTIMIZER=dict(type='RMSprop', lr=1e-3, warmup_steps=100, step_epoch=[170, 200]),
    # OPTIMIZER=dict(type='SGD', lr=2e-5, warmup_steps=100, step_epoch=[170, 200]),

    LOSS=dict(
        type='srhandnetloss',
        auto_weight=False,
        loss_weight=[0.2, 0.2, 0.3, 1.0]   # 四个输出的权重
    )
)

def _get_cfg():
    return cfg