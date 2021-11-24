import matplotlib.pyplot as plt
from data import ZHhand_crop_loader
from utils.visualization_tools import draw_heatmaps, draw_region_maps
from config.resnet_cfg import config_dict
import os
import numpy as np


def test_image():
    """
    show the effect of data augmentations and
    the location of bounding box as well as the central point.
    :return: None
    """
    # draw box and center point on images:
    for i, image in enumerate([img1, img2]):
        plt.subplot(1, 2, i + 1)
        c_x, c_y, w, h = bbox[i].tolist()
        # draw central point
        plt.scatter(c_x, c_y, s=50, c="green", marker='o', alpha=1)
        # draw bounding box
        ax = plt.gca()
        box = plt.Rectangle((c_x - w / 2, c_y - h / 2), width=w, height=h,
                            color="green", fill=False, linewidth=1)
        ax.add_patch(box)
        plt.imshow(image)
    plt.show()


def test_keypoints():
    # (batch, n_joints, h, w)
    kpts = keypoints.squeeze().numpy().astype(np.int32)
    x, y = kpts[:, 0], kpts[:, 1]
    plt.scatter(x, y, s=50, c="green", marker='o', alpha=1)
    plt.imshow(img1)
    plt.show()


def check_heatmap():
    print("-" * 50)
    print(f"{len(target)=}, each target heatmap:")
    for hm in target:
        print(f"{hm.shape=}")
    print("-" * 50)
    print("draw region maps:")
    draw_region_maps(region_maps)
    print("draw keypoints maps:")
    draw_heatmaps(keypoints_maps, k=4)


def check_data_from_hm():
    print("get center, width and height from region maps:")
    batch_cs = dataset.cs_from_region_map(region_maps, k=1, thr=0.8)
    print(f"{batch_cs=}")


def check_pck():
    pred_target = last_hm.clone()
    pred_target[:, 5, 25, 2] += 1.1
    mPCK = dataset.evaluate_pck(pred_keypoints_hm=pred_target, gt_keypoints_hm=last_hm, bbox=bbox)
    print(f"{mPCK=}")  # correct value is between 0 and 1


if __name__ == '__main__':
    # define global variables
    cwd = os.path.abspath('.')
    data_root = config_dict["data_root"]
    test_file = config_dict["train_json"]
    train_file = config_dict["test_json"]
    dataset, test_loader = ZHhand_crop_loader(batch_size=1, num_workers=2).train(img_root=data_root,
                                                                                 ann_file=train_file)
    for i, (img, target, target_weight, label, bbox, keypoints) in enumerate(test_loader):
        if i > 100:
            break
        print(f"{img.shape=}")
        print(f"{bbox=}")
        print(f"{target_weight[0].shape=}")
        # img = img.squeeze(dim=0).numpy()
        img = img.permute(0, 2, 3, 1)
        img1 = img[0].squeeze(dim=0).numpy().clip(0, 1)
        # img2 = img[1].squeeze(dim=0).numpy().clip(0, 1)
        print(f"{img1.shape}")
        # get diverse heatmaps
        last_hm = target[-1]
        region_maps = last_hm[:, :3]
        keypoints_maps = last_hm[:, 4:]  # 1 background + 21 keypoints

        print(f"{keypoints.shape=}")

        # test functions:

        # test_image()
        # check_heatmap()
        # check_data_from_hm()
        # check_pck()
        test_keypoints()
