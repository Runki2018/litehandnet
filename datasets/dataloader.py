import os
import addict
from torch.utils.data import DataLoader, distributed, BatchSampler
from .build_dataset import build_dataset


def make_dataloader(cfg, data_type, use_cpu=False):
    """ 创建数据加载器

    Args:
        cfg (_type_): 读入的配置文件
        is_train (bool, optional): 决定loader的属性和是否启用数据加强. Defaults to True.
        use_cpu(bool): use_cpu=True 是用来验证数据处理是否正确。 

    Returns:
        _type_: _description_
    """
    if not isinstance(cfg, addict.Dict):
        cfg = addict.Dict(cfg)  # 键可以用 . 索引

    dataset = build_dataset(cfg, data_type=data_type)
    batch_per_gpu = cfg.TRAIN.batch_per_gpu
    num_wokers = min([os.cpu_count(),
                        batch_per_gpu if batch_per_gpu > 1 else 0,
                        cfg.TRAIN.workers])
    # num_wokers = 1  # 方便调试
    print(f"{num_wokers=}")

    shuffle = True if data_type == 'train' else False
    num_gpus = cfg.TRAIN.get('num_gpus', 1)
    if not use_cpu:
        images_per_epoch = batch_per_gpu * num_gpus
        # 给每个 rank 对于的进程分配训练样本索引
        sampler = distributed.DistributedSampler(dataset, shuffle=shuffle)
        # 将样本索引每batch_size个元素组成一个list
        batch_sampler = BatchSampler(sampler, images_per_epoch, drop_last=False)

        data_loader = DataLoader(
            dataset,
            # batch_size=images_per_epoch,
            # shuffle=shuffle,
            # sampler=sampler,
            # collate_fn=
            num_workers=num_wokers,
            pin_memory=cfg.TRAIN.pin_memory,
            batch_sampler=batch_sampler,
        )
    else:
        data_loader = DataLoader(
            dataset,
            # batch_size=batch_per_gpu,
            batch_size=64,
            shuffle=shuffle,
            num_workers=num_wokers,
            pin_memory=cfg.TRAIN.pin_memory,
        )

    return dataset, data_loader

