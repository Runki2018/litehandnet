import numpy as np

"""
    MMPose 源码， 需要注意的是对于人体关键点,镜像反转后关键点的含义也会发生变化，
    如右手点变成左手点, 因此除了计算反转后的坐标外,还需要更改代表关键点含义通道顺序。
    对于手部关键点而言, 水平反转前后关键点的含义不会发生变化,但是手的偏向性handerness发生变化,
    即左手变右手,右手变左手。
"""


class TopDownRandomFlip:
    """Data augmentation with random image flip.

    Required keys: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'ann_info'.

    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'flipped'.

    Args:
        flip (bool): Option to perform random flip.
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        center = results['center']

        # A flag indicating whether the image is flipped,
        # which can be used by child class.
        flipped = False
        if np.random.rand() <= self.flip_prob:
            flipped = True
            if not isinstance(img, list):
                img = img[:, ::-1, :]
            else:
                img = [i[:, ::-1, :] for i in img]
            if not isinstance(img, list):
                joints_3d, joints_3d_visible = fliplr_joints(
                    joints_3d, joints_3d_visible, img.shape[1],
                    results['ann_info']['flip_pairs'])
                center[0] = img.shape[1] - center[0] - 1
            else:
                joints_3d, joints_3d_visible = fliplr_joints(
                    joints_3d, joints_3d_visible, img[0].shape[1],
                    results['ann_info']['flip_pairs'])
                center[0] = img[0].shape[1] - center[0] - 1

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['center'] = center
        results['flipped'] = flipped

        return results


def fliplr_joints(joints_3d, joints_3d_visible, img_width, flip_pairs):
    """Flip human joints horizontally.
        
    Note:
        - num_keypoints: K

    Args:
        joints_3d (np.ndarray([K, 3])): Coordinates of keypoints.
        joints_3d_visible (np.ndarray([K, 1])): Visibility of keypoints.
        img_width (int): Image width.
        flip_pairs (list[tuple]): Pairs of keypoints which are mirrored
            (for example, left ear and right ear).

    Returns:
        tuple: Flipped human joints.

        - joints_3d_flipped (np.ndarray([K, 3])): Flipped joints.
        - joints_3d_visible_flipped (np.ndarray([K, 1])): Joint visibility.
    """

    assert len(joints_3d) == len(joints_3d_visible)
    assert img_width > 0

    joints_3d_flipped = joints_3d.copy()
    joints_3d_visible_flipped = joints_3d_visible.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        joints_3d_flipped[left, :] = joints_3d[right, :]
        joints_3d_flipped[right, :] = joints_3d[left, :]

        joints_3d_visible_flipped[left, :] = joints_3d_visible[right, :]
        joints_3d_visible_flipped[right, :] = joints_3d_visible[left, :]

    # Flip horizontally
    joints_3d_flipped[:, 0] = img_width - 1 - joints_3d_flipped[:, 0]
    joints_3d_flipped = joints_3d_flipped * joints_3d_visible_flipped

    return joints_3d_flipped, joints_3d_visible_flipped


class HandRandomFlip(TopDownRandomFlip):
    """Data augmentation with random image flip. A child class of
    TopDownRandomFlip.

    Required keys: 'img', 'joints_3d', 'joints_3d_visible', 'center',
    'hand_type', 'rel_root_depth' and 'ann_info'.

    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center',
    'hand_type', 'rel_root_depth'.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        # base flip augmentation
        super().__call__(results)  # 调用TopDownRandomFlip, 对关键点进行了反转

        # flip hand type, 水平翻转后，手的偏向性发生改变，左右手互换
        hand_type = results['hand_type']
        flipped = results['flipped']
        if flipped:
            hand_type[0], hand_type[1] = hand_type[1], hand_type[0]

        results['hand_type'] = hand_type
        return results


