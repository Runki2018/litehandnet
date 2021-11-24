import numpy as np
from config.config import config_dict
from data import get_dataset
from config.config import DATASET


def _k_means(relative_sizes):
    k = 4  # 边界框相对大小分为四种
    centers = np.array([0.2, 0.4, 0.6, 0.8])  # 初始中心值
    centers_new = centers.copy()
    sum_sizes = np.zeros(4)
    count = np.ones(4)  # 用来统计各类中的个数

    while True:
        for size in relative_sizes:
            index = np.abs(centers - size).argmin()
            sum_sizes[index] += size
            count[index] += 1
        centers_new = sum_sizes / count

        if (centers_new.sum() - centers.sum()) == 0:
            break
        sum_sizes[:] = 0
        count[:] = 1
        centers = centers_new.copy()

    return centers_new, count


def _get_threshold(relative_sizes, cluster_centers):
    thresholds = []

    for i in range(len(cluster_centers) - 1):
        thresholds.append((cluster_centers[i + 1] + cluster_centers[i]) / 2)

    count = []
    n_thr = len(thresholds)
    rs = np.array(relative_sizes)
    count.append(rs[rs < thresholds[0]].size)
    for i in range(1, n_thr):
        count.append(rs[(rs >= thresholds[i - 1]) & (rs < thresholds[i])].size)
    count.append(rs[rs >= thresholds[n_thr-1]].size)

    return thresholds, count


if __name__ == '__main__':
    image_size = config_dict["image_size"]
    image_area = image_size[0] * image_size[1]

    print("preparing {} data...".format(DATASET["name"]))
    _, train_loader = get_dataset('train')
    print("done!")

    sizes_list = []
    if DATASET['name'] in ['zhhand', 'zhhand20', 'freihand', 'freihand100']:
        for img, target, label, bbox in train_loader:
            print(f"{bbox.shape=}")
            bbox_area = (bbox[..., 2] * bbox[..., 3]).item()
            print(f"{bbox_area/image_area =}")
            sizes_list.append(bbox_area / image_area)
    elif DATASET['name'] == 'mpii':
        for _, _, _, bbox, _ in train_loader:
            print(f"{bbox.shape=}")
            bbox_area = (bbox[..., 2] * bbox[..., 3]).item()
            print(f"{bbox_area/image_area =}")
            sizes_list.append(bbox_area / image_area)

    c, n = _k_means(sizes_list)
    print(f"{c=}")
    print(f"{n=}")

    thr_list, n_bbox = _get_threshold(sizes_list, c)
    print(f"{thr_list=}")
    print(f"{n_bbox=}")
