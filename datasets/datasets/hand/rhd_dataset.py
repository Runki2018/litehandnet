# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import os.path as osp
from collections import OrderedDict

import numpy as np
from datasets.base_dataset import Kpt2dDataset
from datasets.dataset_info.dataset_configs import rhd2d_info

class RHD2dDataset(Kpt2dDataset):
    """Rendered Handpose Dataset for top-down 2D hand pose estimation.

    "Learning to Estimate 3D Hand Pose from Single RGB Images",
    ICCV'2017.
    More details can be found in the `paper
    <https://arxiv.org/pdf/1705.01389.pdf>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Rhd keypoint indexes::

        0: 'wrist',
        1: 'thumb4',
        2: 'thumb3',
        3: 'thumb2',
        4: 'thumb1',
        5: 'forefinger4',
        6: 'forefinger3',
        7: 'forefinger2',
        8: 'forefinger1',
        9: 'middle_finger4',
        10: 'middle_finger3',
        11: 'middle_finger2',
        12: 'middle_finger1',
        13: 'ring_finger4',
        14: 'ring_finger3',
        15: 'ring_finger2',
        16: 'ring_finger1',
        17: 'pinky_finger4',
        18: 'pinky_finger3',
        19: 'pinky_finger2',
        20: 'pinky_finger1'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 data_cfg,
                 pipeline,
                 data_type='train'):

        super().__init__(
            data_cfg,
            pipeline,
            data_type,
            dataset_info=rhd2d_info)

        self.ann_info['use_different_joint_weights'] = False
        self.image_size = data_cfg.image_size
        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info.num_joints
        for img_id in self.img_ids:

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            for obj in objs:
                if max(obj['keypoints']) == 0:
                    continue
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                joints_3d[:, :2] = keypoints[:, :2]
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                # the ori image is 224x224
                center, scale = self._xywh2cs(*obj['bbox'][:4], padding=1.25)

                image_file = osp.join(self.img_prefix, self.id2name[img_id])
                gt_db.append({
                    'image_file': image_file,
                    'center': center,
                    'scale': scale,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': np.array(obj['bbox'], np.float32),
                    'bbox_score': 1,
                    'bbox_id': bbox_id
                })
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    def evaluate(self, results, res_folder=None, metric='PCK', **kwargs):
        """Evaluate freihand keypoint results. The pose prediction results will
        be saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['training/rgb/\
                    00031426.jpg']
                - output_heatmap (np.ndarray[N, K, H, W]): model outputs.
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed.
                Options: 'PCK', 'AUC', 'EPE'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'AUC', 'EPE']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = []
        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]

                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value
