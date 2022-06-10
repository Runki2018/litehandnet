# Copyright (c) OpenMMLab. All rights reserved.
import copy
import cv2
import numpy as np

class TopDownGenerateTarget:
    """Generate the target heatmap.

    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info'.

    Modified keys: 'target', and 'target_weight'.

    Args:
        sigma: Sigma of heatmap gaussian for 'MSRA' approach.
        kernel: Kernel of heatmap gaussian for 'Megvii' approach.
        encoding (str): Approach to generate target heatmaps.
            Currently supported approaches: 'MSRA', 'Megvii', 'UDP'.
            Default:'MSRA'
        unbiased_encoding (bool): Option to use unbiased
            encoding methods.
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        keypoint_pose_distance: Keypoint pose distance for UDP.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
        target_type (str): supported targets: 'GaussianHeatmap',
            'CombinedTarget'. Default:'GaussianHeatmap'
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self,
                 sigma=2,
                 kernel=(11, 11),
                 target_type='GaussianHeatmap',
                 encoding='MSRA',
                 unbiased_encoding=False):
        self.sigma = sigma
        self.unbiased_encoding = unbiased_encoding
        self.kernel = kernel
        self.target_type = target_type
        self.encoding = encoding

    def _msra_generate_target(self, cfg, joints_3d, joints_3d_visible, sigma):
        """Generate the target heatmap via "MSRA" approach.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            sigma: Sigma of heatmap gaussian
        Returns:
            tuple: A tuple containing targets.

            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        target = np.zeros((num_joints, H, W), dtype=np.float32)

        # 3-sigma rule
        tmp_size = sigma * 3

        if self.unbiased_encoding:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                # Check that any part of the gaussian is in-bounds
                ul = [mu_x - tmp_size, mu_y - tmp_size]
                br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] == 0:
                    continue

                x = np.arange(0, W, 1, np.float32)
                y = np.arange(0, H, 1, np.float32)
                y = y[:, None]

                if target_weight[joint_id] > 0.5:
                    target[joint_id] = np.exp(-((x - mu_x)**2 +
                                                (y - mu_y)**2) /
                                              (2 * sigma**2))
        else:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
    
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] > 0.5:
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, None]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized, we want the center value to equal 1
                    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], W)
                    img_y = max(0, ul[1]), min(br[1], H)

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight


    def _udp_generate_target(self, cfg, joints_3d, joints_3d_visible, sigma):
        """Generate the target heatmap via 'UDP' approach. Paper ref: Huang et
        al. The Devil is in the Details: Delving into Unbiased Data Processing
        for Human Pose Estimation (CVPR 2020).

        Note:
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W
            - num target channels: C
            - C = K if target_type=='GaussianHeatmap'
            - C = 3*K if target_type=='CombinedTarget'

        Args:
            cfg (dict): data config
            joints_3d (np.ndarray[K, 3]): Annotated keypoints.
            joints_3d_visible (np.ndarray[K, 3]): Visibility of keypoints.
            sigma (float): kernel factor for GaussianHeatmap target
        Returns:
            tuple: A tuple containing targets.

            - target (np.ndarray[C, H, W]): Target heatmaps.
            - target_weight (np.ndarray[K, 1]): (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        heatmap_size = cfg['heatmap_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d_visible[:, 0]

        # Gaussian Heatmap
        target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                            dtype=np.float32)

        tmp_size = sigma * 3

        # prepare for gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, None]

        for joint_id in range(num_joints):
            feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
            mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            mu_x_ac = joints_3d[joint_id][0] / feat_stride[0]
            mu_y_ac = joints_3d[joint_id][1] / feat_stride[1]
            x0 = y0 = size // 2
            x0 += mu_x_ac - mu_x
            y0 += mu_y_ac - mu_y
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
 

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def __call__(self, results):
        """Generate the target heatmap."""
        assert self.encoding in ['MSRA', 'UDP']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        
        if self.encoding == 'MSRA':
            if isinstance(self.sigma, list):
                num_sigmas = len(self.sigma)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                heatmap_size = cfg['heatmap_size']

                target = np.empty(
                    (0, num_joints, heatmap_size[1], heatmap_size[0]),
                    dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                
                for i in range(num_sigmas):
                    target_i, target_weight_i = self._msra_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.sigma[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._msra_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.sigma)

        elif self.encoding == 'UDP':
            if isinstance(self.sigma, list):
                num_factors = len(self.sigma)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                W, H = cfg['heatmap_size']
                target = np.empty((0, num_joints, H, W), dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                
                for i in range(num_factors):
                    target_i, target_weight_i = self._udp_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.sigma[i])
                    
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._udp_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible, self.sigma)
        else:
            raise ValueError(
                f'Encoding approach {self.encoding} is not supported!')

        results['target'] = target
        results['target_weight'] = target_weight

        return results


class SRHandNetGenerateTarget(TopDownGenerateTarget):
    def __init__(self,
                 pred_bbox=True,
                sigma=[2, 2, 2, 2],
                kernel=(11, 11),
                target_type='GaussianHeatmap',
                encoding='MSRA',
                unbiased_encoding=False):

        super().__init__(
            sigma=sigma,
            kernel=kernel,
            target_type=target_type,
            encoding=encoding,
            unbiased_encoding=unbiased_encoding
        )
        self.pred_bbox = pred_bbox
    
    def _region_generate_target(self, bbox, cfg, sigma):
        """ generate region map: [center, w, h]
        Args:
            bbox (np.ndarray): [lx, ly, w, h]
            cfg (dict): 
            sigma (int): default to 2
        """
        center = bbox[:2] + bbox[2:] / 2
        wh = bbox[2:]
        heatmap_size = cfg['heatmap_size']
        image_size = cfg['image_size']
        region_map = np.zeros((3, heatmap_size[1], heatmap_size[0]), np.float32)
        region_weight = np.ones((3, 1))
        
        # classification => center point heatmap
        center_joints = np.array([[center[0], center[1], 1]])  # [1, 3]
        center_vis = np.ones((1, 3))
        
        if self.encoding == 'MSRA':
            target, _ = self._msra_generate_target(
                cfg, center_joints, center_vis, sigma
            )
        elif self.encoding == 'UDP':
            target, _ = self._udp_generate_target(
                         cfg, center_joints, center_vis, sigma
            )

        region_map[:1] = target

        # regression => width&height heatmap
        gamma_x, gamma_y = wh / image_size
        gamma_x = np.clip(gamma_x, 0, 1)  # 0 <= x <= 1
        gamma_y = np.clip(gamma_y, 0, 1)  # 0 <= y <= 1

        feature_stride = heatmap_size / image_size  # [22, 22,22,44,88] / 256
        x, y = center * feature_stride
        tmp_size = 2  # square 5x5 according to SRHandNet
        ul = [int(x - tmp_size), int(y - tmp_size)]
        br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]
        # heatmap range
        x1, x2 = max(0, ul[0]), min(br[0], heatmap_size[0])  # 0~brx, ulx~br, ulx~w_hm
        y1, y2 = max(0, ul[1]), min(br[1], heatmap_size[1])
        region_map[1, y1:y2, x1:x2] = gamma_x
        region_map[2, y1:y2, x1:x2] = gamma_y

        return region_map, region_weight
        
    
    def __call__(self, results):
        """Generate the target heatmap."""
        assert self.encoding in ['MSRA', 'UDP']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        num_sigmas = len(self.sigma)
        cfg = results['ann_info']
        heatmap_size = cfg['heatmap_size']  # [[16, 16],[16, 16],[32, 32],[64, 64]]
        assert len(heatmap_size) == len(self.sigma)

        target_list = []
        target_weight_list = []
        if self.encoding == 'MSRA':
            for i in range(num_sigmas):
                local_cfg = copy.deepcopy(cfg)
                local_cfg['heatmap_size'] = heatmap_size[i]
                heatmap, heatmap_weight = self._msra_generate_target(
                    local_cfg, joints_3d, joints_3d_visible, self.sigma[i])

                if self.pred_bbox:
                    local_cfg['num_joints'] = 1
                    region_map, region_weight = self._region_generate_target(
                        results['bbox'], local_cfg, self.sigma[i])

                    target_list.append(
                        np.concatenate([heatmap, region_map], axis=0))
                    target_weight_list.append(
                        np.concatenate([heatmap_weight, region_weight], axis=0))
                else:
                    target_list.append(heatmap)
                    target_weight_list.append(heatmap_weight)

        elif self.encoding == 'UDP':
            if isinstance(self.sigma, list):
                for i in range(num_sigmas):
                    local_cfg = copy.deepcopy(cfg)
                    local_cfg['heatmap_size'] = heatmap_size[i]
                    heatmap, heatmap_weight = self._udp_generate_target(
                        local_cfg, joints_3d, joints_3d_visible, self.sigma[i])
                    
                    if self.pred_bbox:
                        local_cfg['num_joints'] = 1
                        region_map, region_weight = self._region_generate_target(
                                results['bbox'], local_cfg, self.sigma[i])
                        target_list.append(
                            np.concatenate([heatmap, region_map], axis=0))
                        target_weight_list.append(
                            np.concatenate([heatmap_weight, region_weight], axis=0))
                    else:
                        target_list.append(heatmap)
                        target_weight_list.append(heatmap_weight)
        else:
            raise ValueError(
                f'Encoding approach {self.encoding} is not supported!')

        results['target'] = target_list
        results['target_weight'] = target_weight_list
        return results