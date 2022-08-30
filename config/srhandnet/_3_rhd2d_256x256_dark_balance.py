# w128 best_encoding + SimDR
cfg = dict(
    ID=3,
    MODEL=dict(
        name='srhandnet',
        output_channel=21,  # num_joints + 3 region map
        pred_bbox=False,          # 模型是否预测边界框, 是则不进行旋转变换
        ),

    DATASET=dict(
        name='rhd',    # datasets.datasets.__init__
        num_joints=21,
        image_size=[256, 256],
        heatmap_size=[[16, 16], [16, 16], [32, 32], [64, 64]],

        train=dict(
            ann_file='data/handset/RHD/annotations/rhd_train.json',
            img_prefix='data/handset/RHD/'),
        val=dict(
            ann_file='data/handset/RHD/annotations/rhd_test.json',
            img_prefix='data/handset/RHD/'),
        test=dict(
            ann_file='data/handset/RHD/annotations/rhd_test.json',
            img_prefix='data/handset/RHD/')
        ),

    PIPELINE=dict(
        # TopDownRandomFlip
        flip_prob=0.5,
        # TopDownGetRandomScaleRotation
        rot_prob=0,                     # 在目标检测中或带有region map 不进行旋转操作 0
        rot_factor=0,                   # 在目标检测或带有region map 不进行旋转操作 0
        scale_factor=0.3,
        # TopDownAffine
        use_udp=False,                   # 无偏数据处理
        # TopDownGenerateTarget
        sigma=[2, 2, 2, 2],              # 根据SRHandNet的训练配置, default = 3
        kernel=(11, 11),                 # MSRA unbias编码时用到，sigma=2 => kernel=11
        encoding='MSRA',                 # MSRA | UDP
        unbiased_encoding=False,         # DARK中的编码方法， 在MSRA生成热图时用到，整张热图都生成高斯值
        target_type='GaussianHeatmap',   # 用到不到，默认Gaussian就好
        # !(int) simdr表征的放大倍率。 0, 1, 2, 3, 表示不使用Simdr
        simdr_split_ratio=0,             
    ),

    CHECKPOINT=dict(interval=10, resume=True,
                    load_best=True, save_root='checkpoints/'),

    EVAL=dict(interval=1,
              metric=['PCK', 'AUC', 'EPE'],
              save_best='PCK',
              pck_threshold=0.2),

    TRAIN=dict(
        distributed=True,
        pin_memory=False,
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        find_unused_parameters=False,
        workers=4,
        syncBN=False,
        total_epoches=210,
        batch_per_gpu=16,   # batch_size
    ),

    # 'Adam', 'SGD'
    OPTIMIZER=dict(type='Adam', lr=5e-4, warmup_steps=210, step_epoch=[170, 200]),

    LOSS=dict(
        type='srhandnetloss',
        auto_weight=False,
        loss_weight=[0.3, 0.3, 0.5, 1.0]   # 四个输出的权重
    )
)

def _get_cfg():
    return cfg