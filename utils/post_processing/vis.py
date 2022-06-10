# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import math
import cv2
import numpy as np
import torch
import torchvision


def add_joints(image, joints, color, dataset='COCO'):
    part_idx = VIS_CONFIG[dataset]['part_idx']
    part_orders = VIS_CONFIG[dataset]['part_orders']

    def link(a, b, color):
        if part_idx[a] < joints.shape[0] and part_idx[b] < joints.shape[0]:
            jointa = joints[part_idx[a]]
            jointb = joints[part_idx[b]]
            if jointa[2] > 0 and jointb[2] > 0:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                    color,
                    2
                )

    # add joints
    for joint in joints:
        if joint[2] > 0:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, 2)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1], color)

    return image


def add_bboxes(image, bbox, color):
    H, W, C = image.shape
    x1, y1 = bbox[0, :2]
    x2, y2 = bbox[1, :2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    leftTop = (int(x1), int(y1))  # 左上角的点坐标 (x,y)
    rightBottom = (int(x2), int(y2))  # 右下角的点坐标= (x+w,y+h)
    thickness = 4
    lineType = 8
    image = cv2.rectangle(image, leftTop, rightBottom, color, thickness, lineType)
    return image

def save_valid_image(image, joints, file_name, dataset='COCO', bboxes=None):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for idx, person in enumerate(joints):
        color = [int(i) for i in np.random.randint(0, 255, size=3)]
        add_joints(image, person, color, dataset=dataset)
        if bboxes is not None:
            image = add_bboxes(image, bboxes[idx], color)

    cv2.imwrite(file_name, image)


def make_heatmaps(image, heatmaps):
    heatmaps = heatmaps.mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .cpu().numpy()

    num_joints, height, width = heatmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap*0.7 + image_resized*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid


def draw_bbox(image, centermap, ltrbmap, thr, k=3):
    c, h, w = centermap.shape
    top_v, top_i = torch.topk(centermap.reshape(-1), k=k)
    x = top_i % w  # [k]
    y = torch.div(top_i, w, rounding_mode='trunc')
    # (k, 3)
    coors = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),top_v.unsqueeze(-1)], dim=-1)
    
    for ki in range(k):
        x, y, conf = coors[ki].cpu().tolist()
        if conf < thr:
            continue
        l, t, r, b = ltrbmap[:4, int(y + 0.5), int(x + 0.5)].tolist()
        x1 = x - l * w
        y1 = y - t * h
        x2 = x + r * w
        y2 = y + b * h

        color = [int(i) for i in np.random.randint(0, 255, size=3)]
        image = add_bboxes(image, np.array([[x1, y1], [x2, y2]]), color)
        image = cv2.circle(image, (int(x), int(y)), 2, color, 2)
    
    return image

def make_regionmaps(image, regionmaps, thr=None):
    num_joints, height, width = regionmaps.shape
    # if thr is not None:
    #     # 查看经过nms和均值化后的region map
    #     max_pool = torch.nn.MaxPool2d(5, 1, 2)
    #     mask = max_pool(regionmaps[-1:][None]).eq(regionmaps[-1:][None])
    #     centermap1 = regionmaps[-1:] * mask[0] 
    #     centermap2 = (centermap1 > thr) * 1

    image_resized = cv2.resize(image, (int(width), int(height)))
    # image_resized= draw_bbox(image_resized, centermap1, regionmaps[1:], thr)
    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)
    
    min = float(regionmaps.min())
    max = float(regionmaps.max())
    regionmaps = regionmaps.add(-min).div(max - min + 1e-5)\
                            .mul(255).clamp(0, 255)\
                            .byte().cpu().numpy()

    for j in range(num_joints):
        regionmap = regionmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(regionmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap*0.7 + image_resized*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized
    return image_grid

def make_tagmaps(image, tagmaps):
    num_joints, height, width = tagmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        tagmap = tagmaps[j, :, :]
        min = float(tagmap.min())
        max = float(tagmap.max())
        tagmap = tagmap.add(-min)\
                       .div(max - min + 1e-5)\
                       .mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .cpu().numpy()     

        colored_tagmap = cv2.applyColorMap(tagmap, cv2.COLORMAP_JET)
        image_fused = colored_tagmap*0.9 + image_resized*0.1

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(
                        ndarr,
                        (int(joint[0]), int(joint[1])),
                        2,
                        [255, 0, 0],
                        2
                    )
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_maps(
        batch_image,
        batch_maps,
        batch_mask,
        file_name,
        map_type='heatmap',
        normalize=True,
        detection_threshold=None
):
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_maps.size(0)
    num_channel = batch_maps.size(1)
    map_height = batch_maps.size(2)
    map_width = batch_maps.size(3)
    
    num_map = 1 + num_channel  # 一张原图 + 其余通道

    grid_image = np.zeros(
            (batch_size*map_height, num_map*map_width, 3),
            dtype=np.uint8
        )
 
    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        maps = batch_maps[i]

        if map_type in 'heatmap':
            image_with_hms = make_heatmaps(image, maps)
        elif map_type == 'tagmap':
            image_with_hms = make_tagmaps(image, maps)
        elif map_type == 'regionmap':
             image_with_hms = make_regionmaps(image, maps, detection_threshold)

        height_begin = map_height * i
        height_end = map_height * (i + 1)

        grid_image[height_begin:height_end, :, :] = image_with_hms
        if batch_mask is not None:
            mask = np.expand_dims(batch_mask[i].byte().cpu().numpy(), -1)
            grid_image[height_begin:height_end, :map_width, :] = \
                grid_image[height_begin:height_end, :map_width, :] * mask

    cv2.imwrite(file_name, grid_image)


def save_debug_images(
    config,
    batch_images,
    batch_heatmaps,
    batch_masks,
    batch_outputs,
    prefix,
    vectors=None  # GT 宽高热图
):
    if not config.DEBUG.DEBUG:
        return

    num_joints = config.DATASET.NUM_JOINTS
    if vectors is None:
        batch_pred_conf = batch_outputs[:, :num_joints, :, :]
        batch_pred_vectors = batch_outputs[:, num_joints:, :, :]
    else:
        batch_pred_conf = batch_outputs[0]
        batch_pred_vectors = batch_outputs[1]
        batch_pred_vectors[vectors==0] = 0  # 只显示正样本的回归值

    if config.DEBUG.SAVE_HEATMAPS_GT and batch_heatmaps is not None:
        file_name = '{}_hm_gt.jpg'.format(prefix)
        save_batch_maps(
            batch_images, batch_heatmaps, batch_masks, file_name, 'heatmap'
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        file_name = '{}_hm_pred.jpg'.format(prefix)
        save_batch_maps(
            batch_images, batch_pred_conf, batch_masks, file_name, 'heatmap'
        )

    # if config.DEBUG.SAVE_TAGMAPS_PRED:
    #     file_name = '{}_tag_pred.jpg'.format(prefix)
    #     save_batch_maps(
    #         batch_images, batch_pred_vectors, batch_masks, file_name, 'tagmap'
    #     )
    
    if config.DEBUG.SAVE_REGION_GT and vectors is not None:
        file_name = '{}_region_gt.jpg'.format(prefix)
        save_batch_maps(
            batch_images, vectors, batch_masks, file_name, 'regionmap',
            detection_threshold=config.TEST.DETECTION_THRESHOLD
        )

    if config.DEBUG.SAVE_REGION_PRED:
        file_name = '{}_region_pred.jpg'.format(prefix)
        save_batch_maps(
            batch_images, batch_pred_vectors, batch_masks, file_name, 'regionmap',
            # detection_threshold=config.TEST.DETECTION_THRESHOLD
            detection_threshold=None
        )

