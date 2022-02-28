from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

if __name__ == '__main__':
    # convert 0-based index to 1-based index
    # preds = preds[:, :, 0:2] + 1.0
    preds = np.random.randn(2958, 16, 2)
    #
    # if output_dir:
    #     pred_file = os.path.join(output_dir, 'pred.mat')
    #     savemat(pred_file, mdict={'preds': preds})
    #
    # if 'test' in cfg.DATASET.TEST_SET:
    #     return {'Null': 0.0}, 0.0

    SC_BIAS = 0.6
    threshold = 0.5

    # gt_file = os.path.join(cfg.DATASET.ROOT,
    #                        'annot',
    #                        'gt_{}.mat'.format(cfg.DATASET.TEST_SET))

    gt_file = "./annot/gt_valid.mat"
    gt_dict = loadmat(gt_file)
    dataset_joints = gt_dict['dataset_joints']  # ndarray:(1, 16)
    jnt_missing = gt_dict['jnt_missing']  # (16, 2958)
    pos_gt_src = gt_dict['pos_gt_src']  # (16, 2, 2958)
    headboxes_src = gt_dict['headboxes_src']  # (2, 2, 2958)

    pos_pred_src = np.transpose(preds, [1, 2, 0])

    # 获取13个点的int型序号：
    head = np.where(dataset_joints == 'head')[1][0]  # int64:9
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

    jnt_visible = 1 - jnt_missing  # （16， 2958）
    uv_error = pos_pred_src - pos_gt_src  # （16， 2， 2958）
    uv_err = np.linalg.norm(uv_error, axis=1)  # （16， 2958）
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]  # （2， 2958）
    headsizes = np.linalg.norm(headsizes, axis=0)  # （2958，）
    headsizes *= SC_BIAS
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))  # （16， 2958）
    scaled_uv_err = np.divide(uv_err, scale)  # （16， 2958）
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)  # （16， 2958）

    # （16，） [2115 2485 2889 2888 2478 2119 2878 2932 2932 2932 2928 2935 2944 2944, 2932 2909]
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err <= threshold), jnt_visible)   # （16， 2958） value is 0 or 1

    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)  # (16,) 每个点正确的百分比

    # save
    rng = np.arange(0, 0.5 + 0.01, 0.01)  # [0, 0.01, 0.02, ..., 0.5]
    pckAll = np.zeros((len(rng), 16))  # (51, 16)

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                          jnt_visible)
        pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1),
                                 jnt_count)

    PCKh = np.ma.array(PCKh, mask=False)  # (16, )  [0.04728132387706856 0.04024144869215292 0.0 0.0 0.04035512510088781, 0.04719207173194903 0.0 0.0 0.0 0.0 0.0 0.034071550255536626 0.0 0.0 0.0, 0.0]
    # todo 这里是不是错了，应该6、7、8都不算分值， 后面显示的name value里也没有8，即 'neck' 点
    PCKh.mask[6:8] = True  # 6 、 7分别是盆骨点和前胸点,被掩盖掉，值为 ‘--’， 计算时不纳入计算,

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
