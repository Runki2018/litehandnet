import copy
import bisect
import warnings
import addict
import numpy as np
from collections import abc
from mmcv.utils import is_seq_of
from typing import List, Iterable
from torch.utils.data.dataset import Dataset, T_co, IterableDataset
from datasets.data_pipeline import *
from datasets.datasets import *


class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


def _concat_dataset(cfg, default_args=None):
    types = cfg['type']
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    dataset_infos = cfg.get('dataset_info', None)

    num_joints = cfg['data_cfg'].get('num_joints', None)
    dataset_channel = cfg['data_cfg'].get('dataset_channel', None)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy['ann_file'] = ann_files[i]

        if isinstance(types, (list, tuple)):
            cfg_copy['type'] = types[i]
        if isinstance(img_prefixes, (list, tuple)):
            cfg_copy['img_prefix'] = img_prefixes[i]
        if isinstance(dataset_infos, (list, tuple)):
            cfg_copy['dataset_info'] = dataset_infos[i]

        if isinstance(num_joints, (list, tuple)):
            cfg_copy['data_cfg']['num_joints'] = num_joints[i]

        if is_seq_of(dataset_channel, list):
            cfg_copy['data_cfg']['dataset_channel'] = dataset_channel[i]

        datasets.append(build_dataset(cfg_copy, default_args))

    return ConcatDataset(datasets)

def build_dataset(cfg, data_type='train'):
    assert data_type in ['train', 'val', 'test'], f"Error: {data_type=}"
    
    data_cfg = cfg.DATASET
    P = cfg.PIPELINE
    if cfg.MODEL.name == 'srhandnet':
        GenerateTarget = SRHandNetGenerateTarget(cfg.MODEL.pred_bbox,
                                                 P.sigma, P.kernel, P.target_type, P.encoding, P.unbiased_encoding)
    else:
        GenerateTarget = TopDownGenerateTarget(P.sigma, P.kernel, P.target_type, P.encoding, P.unbiased_encoding)

    if data_type == 'train':
        pipeline = Compose([
            LoadImageFromFile(),
            # HandRandomFlip(P.flip_prob),
            TopDownRandomFlip(P.flip_prob),
            TopDownGetRandomScaleRotation(P.rot_factor, P.scale_factor, P.rot_prob),
            TopDownAffine(P.use_udp),
            ToTensor(),
            NormalizeTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            GenerateTarget])
    else:
        pipeline = Compose([
            LoadImageFromFile(),
            ToTensor(),
            NormalizeTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            GenerateTarget])

    if not isinstance(cfg.DATASET, (list, tuple)):
        dataset_cfg = [cfg.DATASET]
    elif isinstance(cfg.DATASET, dict):
        dataset_cfg = cfg.DATASET
    else:
        raise TypeError(f"{type(cfg.DATASET)=}")

    dataset_list = []
    for data_cfg in dataset_cfg:
        dataset = eval(data_cfg.name)(data_cfg, pipeline, data_type)
        dataset_list.append(dataset)

    # return ConcatDataset(dataset_list)
    return dataset_list[-1]

