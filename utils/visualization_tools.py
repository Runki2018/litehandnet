import cv2 as cv
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def draw_heatmaps(hms, title="heatmap", k=2):
    """

    :param k: the number of keypoints which should be displayed
    :param hms: tensor,(batch, n_joints, h, w)
    :param title: title name of figure
    :return:
    """
    batch = hms.shape[0]
    n_joints = hms.shape[1]

    for bi in range(batch):
        for ni in range(n_joints):
            if ni >= k:
                break  # Only display 3 heatmaps on every hand
            hm = hms[bi, ni].tolist()  # (h, w)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.imshow(hm, cmap=plt.cm.hot_r)
            plt.colorbar(im)
            # plt.subplot(1, 1, 1)
            # plt.imshow(hm, cmap=plt.cm.hot_r)
            # plt.clim(vmin=0, vmax=1)
            # plt.colorbar()
            plt.title(title)
            plt.show()
            plt.close()


def draw_region_maps(hms):
    batch = hms.shape[0]
    n_joints = hms.shape[1]  # 3
    assert n_joints == 3, "input region maps' n_joints = {0}".format(n_joints)
    hms = hms.permute(0, 1, 3, 2)  # (batch, n, h, w) -> (batch, n, w, h)
    titles = ["center", "width", "height"]
    for bi in range(batch):
        fig = plt.figure()
        for ni in range(n_joints):
            plt.subplot(1, 3, ni + 1)
            # ax = plt.gca()
            hm = hms[bi, ni].tolist()  # (h, w)
            plt.imshow(hm, cmap=plt.cm.hot_r)
            plt.clim(0, 1)
            # im = ax.imshow(hm, cmap=plt.cm.hot_r)
            # plt.colorbar(im)
            # ax = plt.contourf(hm, cmap=plt.cm.hot_r, levels=np.linspace(0, 1, 100), extend='both')
            plt.title(titles[ni])
        plt.colorbar()
        plt.show()
        plt.close()


def draw_bbox(img, lx, ly, rx, ry, color=(0, 0, 255)):
    """
            画出边界框的位置

        :param img: cv读入的图片
        :param lx: 左上角点的 x 坐标值
        :param ly: 左上角点的 y 坐标值
        :param rx: 右下角点的 x 坐标值
        :param ry: 右下角点的 y 坐标值
        :param color: BGR red default color
        """
    leftTop = (int(lx), int(ly))  # 左上角的点坐标 (x,y)
    rightBottom = (int(rx), int(ry))  # 右下角的点坐标= (x+w,y+h)
    point_color = color  # BGR
    thickness = 2
    lineType = 8
    img = cv.rectangle(img, leftTop, rightBottom, point_color, thickness, lineType)
    return img


def draw_point(img, keypoints):
    """

        @param img: the image read by cv2, (h, w, c)
        @param keypoints: the keypoints is ndarray, [[x0,y0,c0], [x1, y1, c1], ...], (21,3)
        @return: nothing
        """
    points_list = []
    n_kpts = keypoints.shape[0]
    keypoints = keypoints[:, :2].astype(np.int).tolist()  # [[x0,y0], [x1,y1],...]
    for xy in keypoints:
        points_list.append((xy[0], xy[1]))

    color_list = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 140, 0), (0, 0, 255), (0, 255, 0)]  # RGB
    for i, (r, g, b) in enumerate(color_list):
        color_list[i] = (b, g, r)  # BGR
    point_size = 3
    thickness = 8  # 可以为 0 、4、8
    for i in range(n_kpts):
        if i == 0:
            # print(f"{points_list[i]=}")
            img = cv.circle(img, points_list[i], point_size, color_list[i], thickness)
        else:
            index = (i - 1) // 4 + 1
            img = cv.circle(img, points_list[i], point_size, color_list[index], thickness)
    return img


def draw_text(img, text, bbox, color=(0, 0, 255)):
    """

        :param img: cv2 image
        :param text: the text you draw beside the bbox
        :param bbox: the coordinate list [x1,y1,x2,y2], which is the LT/RB point of bbox
        :param color: text color
        :return: nothing
        """
    # h, w, _ = self.img.shape
    lx, ly, rx, ry = bbox
    text_height = 25  # estimate by drawing text on a image
    y = ry + text_height if ly - text_height < 0 else ly
    xy = lx, y

    img = cv.putText(img=img, text=text,
                     org=xy,
                     fontFace=cv.FONT_HERSHEY_SIMPLEX,
                     fontScale=1,
                     color=color,
                     thickness=2)
    return img
