import numpy as np
import torch
from config import config_dict as cfg

image_size = cfg["image_size"]


def validate_label(keypoints: list, size: tuple):
    """
    如果一张图片中有超过4个关键点超出图像边界，则认为该手势类别为‘0-其他’类
    :param keypoints: ndarray (batch,21,3)
    :param size: (width, height) of the image
    :return: if True , the hand posture is set to 0.
    """

    sum_x_lt_0 = (keypoints[:, 0] < 0).sum()  # the number of (x <0)
    sum_x_gt_w = (keypoints[:, 0] > size[0]).sum()
    sum_y_lt_0 = (keypoints[:, 1] < 0).sum()  # less than 0
    sum_y_gt_h = (keypoints[:, 1] > size[1]).sum()  # great than h

    if (sum_x_lt_0 + sum_x_gt_w + sum_y_lt_0 + sum_y_gt_h) > 4:
        # keypoints[:, :2] = 0
        return True

    return False

def validate_keypoints(keypoints, img_size=(256, 256), n_joints=21):
    """
     验证关键点是否可见，如果变换后的关键点超出图像边界，则将vis设置为-1
    :param keypoints: np.ndarray (1, 21, 3)， （x, y , vis）
    :param img_size: the size of image
    :param n_joints: the number of keypoints
    :return: keypoints
    """
    if keypoints.ndim == 3:
        keypoints = keypoints[0]  # (21, 3)
    n, xys = keypoints.shape
    if n != n_joints:
        raise ValueError(f"{n_joints=} != 21")

    w, h = img_size
    for i in range(n_joints):
        x, y = keypoints[i, :2]
        if x > w - 1 or y > h - 1 or x < 0 or y < 0:
            keypoints[i, 2] = -1  # visible flag
    return keypoints[None]  # (1, 21, 3)


def generate_multi_heatmaps(batch_joints, heatmap_size: int, joints_weight, heatmap_sigma=None, img_size=None):
    """
        用于生成一张图片上有多只手的真值热图
        我移除了joints_vis， 因为可见性一直为1
        :param batch_joints:  [batch, nof_joints, 2 or 3]
        :param heatmap_size:  int, such as 88
        :param heatmap_sigma:  default 3
        :param joints_weight: 关键点权重，决定一个关键点的重要性，如果不可见则为0
        :param img_size: (int or None) 输入图像的大小
        :return: target, target_weight(1: visible, 0: invisible)
    """
    n_hand, n_joints, last_dim = batch_joints.shape  # 因为当前测试的是单手数据集，所以batch数，即手的数目
    
    target = np.zeros((n_hand, n_joints, heatmap_size, heatmap_size), dtype=np.float32)
    target_weight = np.ones((n_hand, n_joints, 1), dtype=np.float32)
    if last_dim == 3:
        vis = batch_joints[:, :, 2] < 1
        if isinstance(vis, torch.Tensor):
            vis = vis.clone().cpu().numpy()
        target_weight[vis] = 0

    img_size = image_size[0] if img_size is None else img_size
    feat_stride = img_size / heatmap_size
    for idx in range(n_hand):
        joints = batch_joints[idx]

        tmp_size = heatmap_sigma * 3

        for joint_id in range(n_joints):
            mu_x = int(joints[joint_id][0] / feat_stride)
            mu_y = int(joints[joint_id][1] / feat_stride)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size), int(mu_y + tmp_size)]
            if ul[0] >= heatmap_size or ul[1] >= heatmap_size \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[idx, joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * heatmap_sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size) - ul[0]  # 0<-ulx~size, 0~size, 0~w_hm-ulx<size
            g_y = max(0, -ul[1]), min(br[1], heatmap_size) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size)  # 0~brx, ulx~br, ulx~w_hm
            img_y = max(0, ul[1]), min(br[1], heatmap_size)

            v = target_weight[idx, joint_id]
            if v > 0.5:
                target[idx, joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if cfg['use_different_joints_weight']:    
            target_weight = np.multiply(target_weight, joints_weight)

    return target, target_weight


def get_bbox(keypoints, alpha=1.3):
    """ 根据关键点获取边界框，alpha是将仅仅贴合关键点的框放大的倍数

    Args:
        keypoints ([type]): numpy.array([n_hand, 21, 3])
        alpha (float, optional): 方法倍数. Defaults to 1.3.

    Returns:
        [numpy.array]: bbox([n_hand, 4]), last axist = (x_center, y_center, w, h)
    """
    n_hand = keypoints.shape[0]
    bbox = np.zeros((n_hand, 4), dtype=np.float32)
    for i_hand in range(n_hand):
        kpts = keypoints[i_hand]
        if kpts.size == 0:
            continue
        x1y1 = kpts.min(0)[:2]  # array([x1, y1, 1])
        x2y2 = kpts.max(0)[:2]  # array([x2, y2, 1])

        s1 = (x2y2 - x1y1)  # (w, h) of the tight box around keypoints
        s2 = s1 * alpha  # (s_w, s_h) of the larger box around keypoints
        x1y1 -= (s2 - s1) / 2
        x2y2 += (s2 - s1) / 2
        x1, y1 = x1y1
        x2, y2 = x2y2
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_size[0], x2), min(image_size[1], y2)
        
        cx = (x1 + x2) / 2  # the center of bbox
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        bbox[i_hand] = np.array([cx, cy, w, h], np.int16)
    return bbox

def get_multi_regions(bbox, heatmap_size, heatmap_sigma=None, img_size=None):
    """

    :param bbox: ndarray (n_hand, 4), (cx, cy, w, h)
    :param heatmap_size: default 88
    :param heatmap_sigma: default 3
    :param alpha: 将由关键点计算所得的bbox缩放alpha倍,
    :param img_size: int or None, 输入图像的大小, 默认为方形图像，所以只需要一边长度
    :return: 包括三个通道的region maps， 计算loss时的通道权重，放缩后新图像大小上bbox
    """
    n_hand = bbox.shape[0]
    # 每个手有3个region maps, 每张热图也有其相应的权重
    multi_hand_regions = np.zeros((n_hand, 3, heatmap_size, heatmap_size), dtype=np.float32)
    region_weights = np.ones((n_hand, 1, 1), dtype=np.float32)

    for i_hand in range(n_hand):
        c = bbox[i_hand, :2]
        s = bbox[i_hand, 2:]

        temp_regions, _ = generate_multi_heatmaps(c.reshape((1, 1, 2)), heatmap_size, region_weights, heatmap_sigma=heatmap_sigma, img_size=img_size)

        multi_hand_regions[i_hand, 0] = temp_regions[0]
        multi_hand_regions[i_hand, 1:] = get_hw_region_map(c, s, heatmap_size, heatmap_sigma, image_size[0] if img_size is None else img_size)

    return multi_hand_regions


def get_hw_region_map(c, s, heatmap_size, heatmap_sigma, img_size):
    """
    :param c: the center point of bbox, ndarray([cx, cy])
    :param s: ndarray([w_box,h_box])
    :param heatmap_size: default 88
    :param heatmap_sigma: default 3 ， 决定了高斯热图的范围
    :param img_size: int or None, 输入图像的大小, 默认为方形图像，所以只需要一边长度
    :return: wh_region_map, (2,h_hm, w_hm)
    """

    wh_region_map = np.zeros((2, heatmap_size, heatmap_size), dtype=np.float32)
    w, h = img_size, img_size  # ! 现在都是方向图片
    gamma_x, gamma_y = s[0] / w, s[1] / h
    gamma_x = np.clip(gamma_x, 0, 1)  # 0 <= x <= 1
    gamma_y = np.clip(gamma_y, 0, 1)  # 0 <= y <= 1

    stride_size = heatmap_size / img_size  # [22, 22,22,44,88] / 256
    x, y = np.array(c) * stride_size
    tmp_size = heatmap_sigma * 3  # 3*sigma_r according to SRHandNet
    ul = [int(x - tmp_size), int(y - tmp_size)]
    br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]
    # heatmap range
    x1, x2 = max(0, ul[0]), min(br[0], heatmap_size)  # 0~brx, ulx~br, ulx~w_hm
    y1, y2 = max(0, ul[1]), min(br[1], heatmap_size)
    wh_region_map[0, y1:y2, x1:x2] = gamma_x
    wh_region_map[1, y1:y2, x1:x2] = gamma_y

    return wh_region_map


def get_mask1(kpts_heatmaps):
    """
        获取背景热图，背景部分为1，有手部关键点的区域小于1
    :param kpts_heatmaps: (n_joints, h_hm, w_hm)
    :return: a background heatmap, (h_hm, w_hm)
    """
    n_joints = kpts_heatmaps.shape[0]
    heatmap_size = kpts_heatmaps.shape[1]
    bg_hm = np.ones((heatmap_size, heatmap_size), dtype=np.float32) * n_joints
    bg_hm = bg_hm - kpts_heatmaps.sum(axis=0)
    # normalized to 1
    bg_min = bg_hm.min()
    bg_max = bg_hm.max()
    bg_hm = (bg_hm - bg_min) / (bg_max - bg_min + 1e-5)

    return bg_hm


def get_mask2(kpts_heatmaps, v1=0.1, v2=0.8):
    """
        获取关键点掩膜热图，背景部分为0，有手部关键点的区域大于0，关键点越密集值越大，最大值为1
        将标准化后处于min_value1~min_value2的值改为value2
    :param kpts_heatmaps: (n_joints, h_hm, w_hm)
    :param v1: default 0.2 直接赋值区域的下界
    :param v2: default 0.5 直接赋值区域的上界
    :return: a background heatmap, (h_hm, w_hm)
    """

    bg_hm = kpts_heatmaps.sum(axis=0)  # (heatmap_size, heatmap_size)
    # normalized to 1
    bg_min = bg_hm.min()
    bg_max = bg_hm.max()
    bg_hm = (bg_hm - bg_min) / (bg_max - bg_min + 1e-5)
    bg_hm[(bg_hm > v1) & (bg_hm < v2)] = v2

    return bg_hm


def get_mask3(kpts_heatmaps):
    """
        获取关键点掩膜热图，背景部分为0，有手部关键点的区域为1，这中掩膜热图可能更有有助于做注意力乘法
    :param kpts_heatmaps: (n_joints, h_hm, w_hm)
    :return: a background heatmap, (h_hm, w_hm)
    """

    bg_hm = kpts_heatmaps.sum(axis=0)  # (heatmap_size, heatmap_size)
    # normalized to 1
    bg_hm[bg_hm > 0.3] = 1
    return bg_hm


def combine_together(hm_input):
    """
    将一张图片上多手的热图，在同一关键点或同一区域图上组合在一起，即一张热图上有多只手的信息，属于bottom-up的方法
    对于两张热图非零重叠的的部分取平均值，对于其中一个热图为零的区域直接相加
    :input_hm: (n_hand, n_joints, hm_size, hm_size)
    :return: (n_joints, hm_size, hm_size)
    """
    n_hand, n_joints, hm_size, hm_size = hm_input.shape
    hm_output = hm_input[0].astype(np.float32)
    factor = np.ones((n_joints, hm_size, hm_size), np.float32)  # 一个非零像素是由多少个热图叠加而成，用于求平均
    if n_hand > 1:
        for i in range(1, n_hand):
            for j in range(n_joints):
                hm1 = hm_input[i, j]
                hm = hm_output[j]  # 与hm_output共享内存，修改值作用到hm_output上
                hm2 = hm.copy()  # 独立比较的矩阵，防止比较过程中值发生变化而导致运算错误
                # 对于两张热图都有非零值的区域，取平均值
                hm[(hm2 > 0) & (hm1 > 0)] += hm1[(hm2 > 0) & (hm1 > 0)]
                factor[j][(hm2 > 0) & (hm1 > 0)] += 1
                # 只有一张热图有非零值的区域，直接相加
                hm[(hm2 == 0) | (hm1 == 0)] += hm1[(hm2 == 0) | (hm1 == 0)]
        hm_output /= factor  # 取平均值

    return hm_output
