# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json_tricks as json
import json
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

from data.mpii.JointsDataset import JointsDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.training_kits import set_seeds
from config import seed, DATASET, config_dict as cfg

# set_seeds(seed)  # 多卡训练随机数种子不能相同，否则数据增广单一


# import logging
# logger = logging.getLogger(__name__)


class MPIIDataset(JointsDataset):
    def __init__(self, root, ann_file, is_train, transform=None):
        super().__init__(is_train, transform)

        self.root = root
        self.ann_file = ann_file
        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.db = self._get_db()

        if is_train and cfg['select_data']:  # False
            self.db = self.select_data(self.db)

        # logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        # file_name = os.path.join(self.root, 'annot', self.ann_file)
        file_name = self.ann_file

        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        i = 0
        num_sample = cfg['num_sample']
        for a in anno:
            i += 1
            if 0 < num_sample < i:
                break  # 先测试一下num_sample个样本的训练效果

            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # todo 这个是这个作者自己加入的规则？相当于放大了框？
            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25


            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            if self.ann_file != 'test.json':
                joints = np.array(a['joints'])  # (16, 2)
                joints[:, 0:2] = joints[:, 0:2] - 1  # 将Matlab下标从1开始，改为python从0开始
                joints_vis = np.array(a['joints_vis'])  # (16,)
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]  # 3d中，joints[:, 3]是预测点的得分？还是为了兼容3d数据集
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db

    def evaluate(self, preds, output_dir, *args, **kwargs):
        """
            preds: np.ndarray[n_images, n_joints, 3]
            output_dit: a path to save the output result
        """
        # convert 0-based index to 1-based index 重新转换成matlab的数组下标形式
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg['test_set']:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6  # 这个偏置的作用？
        threshold = 0.5

        gt_file = os.path.join(DATASET['root'],
                               'annot',
                               'gt_{}.mat'.format(cfg['test_set']))
        gt_dict = loadmat(gt_file)  # ‘annot/gt_valid.mat’
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']


class MPIILoader:
    def __init__(self):
        self.root = DATASET['root']
        self.test_file = DATASET['test_file']
        self.train_file = DATASET['train_file']
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])

    def test(self, just_dataset=False):
        valid_dataset = MPIIDataset(root=self.root, ann_file=self.test_file,
                                    is_train=False, transform=self.transform)
        print("sample number of testing dataset: ", valid_dataset.__len__())
        if just_dataset:
            return valid_dataset
        else:    
            loader = DataLoader(
                valid_dataset,
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=cfg['workers'],
                # pin_memory=cfg['pin_memory'],
                pin_memory=False,
                drop_last=False,
            )
            return valid_dataset, loader

    def train(self, just_dataset=False):
        train_dataset = MPIIDataset(root=self.root, ann_file=self.train_file,
                                    is_train=True, transform=self.transform)
        print("sample number of training dataset: ", train_dataset.__len__())
        if just_dataset:
            return train_dataset
        else: 
            loader = DataLoader(
                train_dataset,
                batch_size=cfg["batch_size"],
                shuffle=cfg['train_shuffle'],
                num_workers=cfg['workers'],
                pin_memory=cfg['pin_memory'],
                drop_last=True,
            )
            return train_dataset, loader


