# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from data.handset.dataset_function import *

from config import pcfg, config_dict as cfg

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200  # 行人框：使用center和scale标注，人体尺度关于200像素高度。也就是除过了200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train

        # self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg['data_format']  # ‘jpg'

        self.scale_factor = cfg['scale_factor']  # 0.25
        self.rotation_factor = cfg['rotation_factor']  # 30
        self.flip = cfg['flip']  # True
        self.num_joints_half_body = cfg['num_joints_half_body']  # 8
        self.prob_half_body = cfg['prob_half_body']  # 0.0
        self.color_rgb = cfg['color_rgb']  # 'False'

        self.target_type = cfg['target_type']
        self.image_size = np.array(cfg['image_size'])  # [256, 256]
        self.heatmap_size = np.array(cfg["hm_size"])  # [64, 64]
        self.simdr_split_ratio = cfg['simdr_split_ratio']
        self.sigma = cfg["hm_sigma"][0]  # 2
        self.max_num_object = pcfg["max_num_bbox"]  # 一张图片上最多保留的目标数
        self.use_different_joints_weight = cfg['use_different_joints_wight']  # False
        self.joints_weight = 1
        self.image_width, self.image_height = cfg['image_size']
        self.aspect_ratio = self.image_width * 1.0 / self.image_height

        self.transform = transform
        self.db = []  # 数据集列表，COCO和MPII的实现不同， self.db = self._get_db()

        mask_func = {1: get_mask1, 2: get_mask2, 3: get_mask3}
        self.get_mask = mask_func[cfg["mask_type"]]  # 手部区域部分，要用背景热图，还是手部关键点掩膜

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []  # 9个点
        lower_joints = []  # 7个点
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,  # w = s * self.pixel_std = s * 200
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self, ):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:  # 由于格式是 ’jpg‘ 不是 ’zip‘：执行下面这步
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                    and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor  # 0.25
            rf = self.rotation_factor  # 30
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        # target = torch.from_numpy(target)
        # target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,  # 对于mpii 一直是 ''
            'imgnum': imgnum,  # 0
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }
   
        # if cfg['need_region_map']:
        #     hm_output, bbox = self.get_hm_target(joints, target)
        #     return input, hm_output, target_weight, bbox, meta
        # else:
        #     return input, target, target_weight, meta
        kpts_hm, bbox = self.get_hm_target(joints, target)
        target_x, target_y = self.generate_sa_simdr(joints, target_weight, self.sigma)
        # todo: 多守数据集需要完善这个代码
        gt_kpts = np.zeros((self.max_num_object, self.num_joints, 3))
        gt_kpts[0] = joints   
        
        return input, target_x, target_y, target_weight, kpts_hm, bbox, gt_kpts, meta

    def get_hm_target(self, joints, kpt_target):
        # 热图各个通道的含义 heatmaps = 3 region map [+ 1 background] + 16 keypoints

        # TODO: 现在只有在 self.heatmap_size[1] == self.heatmap_size[0] 时有效
        joints = joints[joints[:, 0] > 0]
        if joints.size == 0:
            joints = np.array([[-1., -1., 0.]])

        bbox = get_bbox(joints[None])
        region_map, _= get_multi_regions(bbox, self.heatmap_size[0], self.sigma)
        region_map = combine_together(region_map)
        mask = self.get_mask(kpt_target)[None]
        hm_output = np.concatenate((region_map, kpt_target, mask), axis=0)
        # hm_output = np.concatenate((region_map, kpt_target), axis=0)
        return hm_output, bbox
    
    def generate_sa_simdr(self, joints, target_weight, sigma):
        """
        :param joints:  [num_joints, 3]
        :param target_weight: [num_joints, 1] (1: visible, 0: invisible)
        :param sigma: 1d gaussion sigma
        :return: target
        """
        # todo: 这个是单手的，还不能多个点叠加到一起
        sigma = sigma if sigma > 2 else 2
        target_x = np.zeros((self.num_joints,
                             int(self.image_size[0] * self.simdr_split_ratio)),
                            dtype=np.float32)
        target_y = np.zeros((self.num_joints,
                             int(self.image_size[1] * self.simdr_split_ratio)),
                            dtype=np.float32)

        for joint_id in range(self.num_joints):
            if target_weight[joint_id] > 0:                   
                mu_x, mu_y = joints[joint_id, :2] * self.simdr_split_ratio
                
                x = np.arange(0, int(self.image_size[0] * self.simdr_split_ratio), 1, np.float32)
                y = np.arange(0, int(self.image_size[1] * self.simdr_split_ratio), 1, np.float32)

                # target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))
                # target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))
                
                target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * sigma ** 2))) 
                target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * sigma ** 2)))
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target_x, target_y

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std ** 2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center - bbox_center), 2)
            ks = np.exp(-1.0 * (diff_norm2 ** 2) / (0.2 ** 2 * 2.0 * area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        target = np.zeros((self.num_joints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight
