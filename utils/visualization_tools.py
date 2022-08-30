import cv2 as cv
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch


def draw_centermap(centermap, title='centermap'):
    """可视化Centermap

    Args:
        centermap (tensor): [1, 5, h, w], batch = 1
        title (str, optional): [description]. Defaults to 'centermap'.
    """
    assert centermap.ndim == 4
    hm = dict(center=centermap[0, 0:1],
            w=centermap[0, 1:2],
            h=centermap[0, 2:3],
            offset_x=centermap[0, 3:4],
            offset_y=centermap[0, 4:5])
    
    out = []
    for i, (name, heatmap) in enumerate(hm.items()):
        heatmap = heatmap.permute(1, 2, 0)
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = cv.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
        out.append(heatmap)
        
    return out


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


def draw_bbox(img, bbox, xyxy=True, color=(0, 0, 255)):
    """
            画出边界框的位置

        :param img: cv读入的图片
        :param bbox: 边界框,左上角和右下角表示xyxy 或 左上角和宽高表示xywh
        :param xyxy: 确定bbox的格式
        :param color: BGR red default color
        """
    if xyxy:
        lx, ly, rx, ry = bbox
    else:
        lx, ly = bbox[0], bbox[1]
        rx, ry = lx + bbox[2], ly + bbox[3]
        
    leftTop = (int(lx), int(ly))  # 左上角的点坐标 (x,y)
    rightBottom = (int(rx), int(ry))  # 右下角的点坐标= (x+w,y+h)
    point_color = color  # BGR
    thickness = 2
    lineType = 8
    img = cv.rectangle(img, leftTop, rightBottom, point_color, thickness, lineType)
    return img


def draw_point(img, keypoints, is_rgb=True):
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
    if not is_rgb:
        for i, (r, g, b) in enumerate(color_list):
            color_list[i] = (b, g, r)  # BGR
    point_size = 2
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
