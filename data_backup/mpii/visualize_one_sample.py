import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
from pathlib import Path
import numpy as np


def show_one_sample(ann_file, idx=0):
    """
        可视化一下每个行人框关键点和用center、scale定义的框。
    :param ann_file:
    :param idx: 单人框序号
    :return:
    """
    ann_file = Path(ann_file)
    if not ann_file.exists():
        raise ValueError('ann_file: {} not exits!'.format(ann_file))
    ann = json.load(open(ann_file, 'r'))

    sample_dict = ann[idx]
    print(f"{sample_dict.keys()=}")
    joints = np.array(sample_dict['joints'])  # (16, 2)
    joints -= 1  # (16, 2)
    c = np.array(sample_dict['center'], dtype=np.int64)
    s = np.array([sample_dict['scale'], sample_dict['scale']], dtype=np.float32)

    ax = plt.gca()
    image = plt.imread('./images/' + sample_dict['image'])

    # 根据关键点算出来的框大小
    x1, y1 = np.min(joints, axis=0)
    x2, y2 = np.max(joints, axis=0)
    c = (x1 + x2) // 2, (y1 + y2)//2
    a = 1.35
    w, h = int((x2 - x1) * a), int((y2 - y1) * a)
    xy = c[0] - w/2, c[1] - h/2

    #  作者的裁剪的框大小
    # if c[0] != -1:
    #     c[1] = c[1] + 15 * s[1]
    #     s = s * 1.25
    # w, h = (s * 200).astype(np.int64)
    # xy = (int(c[0] - w/2), c[1]-h/2)

    print(f"{xy=}, {w=}, {h=}")
    plt.scatter(x=c[0], y=c[1], s=60, alpha=0.8)
    plt.scatter(x=joints[:, 0], y=joints[:, 1], s=60, alpha=0.8)
    bbox = patches.Rectangle(xy=xy, width=w, height=h, color='blue', fill=False, linewidth=2)
    ax.add_patch(bbox)

    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    path = "./annot/train.json"
    show_one_sample(path, idx=30)
