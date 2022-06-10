import time

import numpy as np
import torch
import torchvision
from utils.bbox_metric import xywh2xyxy, box_iou, bbox_iou
from config import pcfg


def evaluate_pck(pred_keypoints_hm, gt_keypoints_hm, bbox, image_size=256, target_weight=None, thr=0.2):
    """
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than y,x locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies

    :param thr: sigma threshold scalar
    :param bbox: bounding box (batch,n_hand, 4)  [c_x, c_y, w, h]
    :param hm_type: default is 2D gaussian
    :param pred_keypoints_hm: batch_heatmaps,numpy.ndarray([batch, n_joints, height, width])
    :param gt_keypoints_hm: batch_heatmaps,numpy.ndarray([batch, n_joints, height, width])
    :param target_weight: batch_heatmaps,numpy.ndarray([batch, n_joints, 1])
    """

    bbox = bbox[:, 0]  # todo 多手的时候怎么重新设计一下计算pck的方法

    batch, n_joints, h, w = pred_keypoints_hm.shape
    gt_keypoints_hm = gt_keypoints_hm.to(pred_keypoints_hm.device)

    pred_coors, _ = get_coordinates_from_heatmap(pred_keypoints_hm)  # (batch, n_joints,xy)
    target_coors, _ = get_coordinates_from_heatmap(gt_keypoints_hm)

    # mapping to original image size
    factor = (torch.tensor(image_size) / torch.tensor([w, h])).to(pred_coors.device)
    pred_coors *= factor
    target_coors *= factor
        # todo: check the value of mPCK
    max_wh, _ = bbox[:, 2:].max(dim=-1)  # (batch,), (batch,)
    max_wh = max_wh.to(pred_keypoints_hm.device)

    if target_weight is None:
        target_weight = torch.ones_like(pred_coors)
    else:
        target_weight = torch.cat([target_weight, target_weight], dim=-1)
    target_weight = target_weight.to(pred_coors.device)

    pck = []
    for i in range(pred_coors.shape[0]):
        pred = pred_coors[i][target_weight[i] == 1].view(-1, 2)  # (n_visable_joints, xy)
        target = target_coors[i][target_weight[i] == 1].view(-1, 2) 

        distance = torch.norm(pred - target, dim=-1) / max_wh[i]

        num_lt_thr = distance < thr  # num_lt_thr.shape = (batch, n_joints)
        PCK = num_lt_thr.sum() / target_weight[i].sum() * 2
        pck.append(PCK.item())
    mPCK = np.mean(pck)

    return mPCK


def get_coordinates_from_heatmap(batch_heatmaps):
    """
    get coordinates (n_joint, xy) = (21, 2) from heatmaps
    :param batch_heatmaps: numpy.ndarray([batch, n_joints, height, width])
    :return:  keypoint coordinates (batch, n_joint, xy) = (batch, 21, 2) and score (batch, 21, 1)
    """
    assert isinstance(batch_heatmaps, torch.Tensor), "batch_heatmaps should be torch.Tensor"
    assert batch_heatmaps.dim() == 4, "batch_heatmaps should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    n_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, n_joints, -1)
    max_vals, idx = torch.max(heatmaps_reshaped, dim=2)

    max_vals = max_vals.unsqueeze(dim=-1)  # (batch,n_joints, 1)
    idx = idx.float()  # (batch, n_joints)

    preds = torch.zeros((batch_size, n_joints, 2), dtype=torch.float).to(batch_heatmaps.device)

    preds[:, :, 0] = idx % width  # x column
    preds[:, :, 1] = torch.floor(idx / width)  # y row

    # to compare the max values with 0
    preds_mask = torch.gt(max_vals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= preds_mask
    return preds, max_vals


# ----------------- region maps 和 计算bbox AP相关 -----------------

def cs_from_region_map(batch_region_maps, image_size=256, k=20, thr=0.8):
    """
        根据置信度选出大于阈值的k个目标候选框。
    :param thr: 置信度阈值，大于该阈值的候选框点，才会计算bbox，不然为[0，0，0，0]
    :param k: k是最大候选框个数, 从中心点热图中选出前K个得分最高的点，作为候选框中心点，并计算以该点为中心的bbox
    :param batch_region_maps: tensor(batch, 3, h_heatmap, w_heatmap)
    :return: the center and size(width, height) of bbox,  (batch, n_hand, 5)
             每个元素是 [center_x, center_y, w, h, confidence]
    """

    batch = batch_region_maps.shape[0]
    heatmap_size = batch_region_maps.shape[-1]
    c_region_maps = batch_region_maps[:, 0].unsqueeze(dim=1).reshape((batch, -1))  # (batch, h*w)
    wh_region_maps = batch_region_maps[:, 1:]  # (batch, 2, h, w)
    # print(f"{c_region_maps.shape=}")
    # print(f"{wh_region_maps.shape=}")
    
    candidates = torch.zeros((batch, k, 5), dtype=torch.float32)

    # get center
    top_val, top_idx = torch.topk(c_region_maps, k=k)  # (batch, k), (batch, k)
    for bi in range(batch):
        flags = top_val[bi] > thr  # tensor, (k,)
        for ki, flag in enumerate(flags):
            if flag:
                # todo 这里用 a // b会有警告，建议替换为torch.div(a, b, rounding_mode='trunc')
                center = top_idx[bi][ki] % heatmap_size, \
                         torch.div(top_idx[bi][ki], heatmap_size, rounding_mode='trunc') # top_idx[bi][ki] // heatmap_size
                # print(f"{center=}")
                candidates[bi, ki, 0] = center[0]  # column, x in heatmap
                candidates[bi, ki, 1] = center[1]  # row, y in heatmap
                s = _get_wh(wh_region_maps, bi, center, image_size)
                candidates[bi, ki, 2] = s[0]
                candidates[bi, ki, 3] = s[1]

    candidates[..., 4] = top_val  # 记录confidence

    # scale center x and y from heatmap to original image
    up_stride = image_size / heatmap_size
    candidates[..., :2] *= up_stride

    return candidates


def _get_wh(wh_region_maps, bi, center, image_size, heatmap_sigma=2):
    # get width and height of bbox
    center = torch.tensor(center)
    heatmap_size = wh_region_maps.shape[-1]

    temp_size = heatmap_sigma * 3  # 9
    ul = center - temp_size  # (2,)
    br = center + temp_size + 1  # (2,)

    # heatmap range
    x1 = ul[0].clip(min=0, max=heatmap_size - 1)  # (1,)
    x2 = br[0].clip(min=0, max=heatmap_size - 1)  # (1,)
    y1 = ul[1].clip(min=0, max=heatmap_size - 1)
    y2 = br[1].clip(min=0, max=heatmap_size - 1)
    # TODO: check the c and s in situation of multiple hands
    # TODO: use NMS to get multiple c ,and then to get s according to c
    gamma_x = wh_region_maps[bi, 0, y1:y2, x1:x2].flatten().mean(dim=-1)
    gamma_y = wh_region_maps[bi, 1, y1:y2, x1:x2].flatten().mean(dim=-1)
    # gamma_x = torch.clip(gamma_x, 0, 1)  # 0 <= x <= 1
    # gamma_y = torch.clip(gamma_y, 0, 1)  # 0 <= y <= 1
    # w = h = image_size  # ! only fit the condition of same size
    # s_x = gamma_x * w
    # s_y = gamma_y * h
    s_x = gamma_x * image_size / heatmap_size
    s_y = gamma_y * image_size / heatmap_size
    return s_x, s_y


def non_max_suppression(prediction, iou_threshold=0.8, conf_threshold=0.8, max_num=100):
    """ 非最大值抑制

    :param prediction: 预测的bbox，(batch, k, 5), (lx,ly, w, h, conf), k个候选框按置信度降序排列
    :param iou_threshold: 大于该阈值的框被认为是同一个目标的框，被抑制掉
    :param conf_threshold: 大于该阈值的框被认为是前景框
    :param max_num: 从候选框中最终筛选出的最大目标框数目
    :Returns: a list of the format: (n_images, n_boxes, 5)， 每张图片的预测框数目可能不同
                [[ [x, y, w, h, conf],  # 第一张图片的第一个预测框
                   [x, y, w, h, conf]， # 第一张图片的第二个预测框
                    ....
                    ]，
                 [ [x, y, w, h, conf],  # 第二张图片的第一个预测框
                   [x, y, w, h, conf]， # 第二张图片的第二个预测框
                    ....
                    ]，
                 ...
                ]
    """

    # Settings
    merge = False  # todo merge for best mAP, 将同一个目标的多个框融合，得到一个与真值框更接近的框
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    output = [None] * prediction.shape[0]  # batch 张图片的预测框，非None则该图片有预测到目标框
    for i, x in enumerate(prediction):
        x = x[x[:, 4] > conf_threshold]  # confidence 根据obj confidence虑除背景目标
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # # width-height 虑除小目标

        if not x.shape[0]:  # 如果 x.shape = [0, 4]， 则当前图片没有检测到目标
            continue

        boxes = xywh2xyxy(x[:, :4])  # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        scores = x[:, 4]
        # NMS

        index = torchvision.ops.nms(boxes, scores, iou_threshold)
        index = index[:max_num]  # 一张图片最多只保留前max_num个目标信息

        output[i] = x[index].tolist()
        if (time.time() - t) > time_limit:
            break  # 超时退出

    return output


def evaluate_ap(batch_region_maps, gt_boxes, image_size, k=20, iou_thr=None):
    """
        输出经过非最大值抑制后的预测框和 ap。

    param max_num: 一张图像经过NMS后能保留的最大预测框数
    param conf_thr:  预测框大于该阈值才被视为正确预测到目标
    param k:  经过NMS前的候选框数目
    param batch_region_maps: 模型输出的region maps热图， a tensor of shape [batch, 3, hm_size, hm_size]
    param nms_thr: 用于非最大值抑制的IOU阈值。
    param iou_thr: if None， iou = 0.5:0.05:0.95， otherwise it should be a specific iou value
    :return: AP and a list of predicted bboxes, [lx, ly, w, h, conf]
    """

    candidates = cs_from_region_map(batch_region_maps, image_size, k, pcfg['detection_threshold'])
    # print(f"{candidates[:20]=}")
    pred_bboxes = non_max_suppression(candidates, pcfg["iou_threshold"],
                                      pcfg['detection_threshold'], pcfg["max_num_bbox"])
    # print(f"{pred_bboxes[0]=}")

    gt_boxes = gt_boxes.tolist() if isinstance(gt_boxes, torch.Tensor) else gt_boxes
    # print(f"{gt_boxes[0]=}")
    ap50, ap = count_ap(pred_boxes=pred_bboxes, gt_boxes=gt_boxes, iou_threshold=iou_thr)
    # print(f"{ap=}")

    return float(ap50), float(ap), pred_bboxes


def count_ap(pred_boxes: list, gt_boxes: list, iou_threshold=None, verbose=False):
    """
        根据PR曲线计算AP， bbox的类别只有手这一类，所以只有目标置信度，没有类别置信度.
        这里的计算方法，采用VOC2010后的计算方法。
        https://www.bilibili.com/video/BV1ez4y1X7g2
    :param pred_boxes: (list), 经过NMS后的预测bbox， (batch, n_hand, 5), (lx, ly, w, h, conf)
    :param gt_boxes:  (list),真值bboxes
    :param iou_threshold:  IOU阈值，大于该阈值为预测真, if None, IOU = 0.5:0.95:0.05
    :param verbose:  是否答应计算表
    :return: p, r, ap
    """
    num_pred = 0  # 预测的目标框数目
    for boxes_list in pred_boxes:
        num_pred += len(boxes_list) if boxes_list is not None else 0

    if num_pred == 0:
        return 0, 0

    num_object = 0  # 真值框数目
    for boxes_list in gt_boxes:
        num_object += len(boxes_list)

    # img_id(0), pred_id(1), gt_id(2), conf(3), hit(4), TP(5), FP(6), FN(7), Precision(8), Recall(9), iou(10)
    # img_id： 图片号
    # pred_id： 第 img_id 张图片上，第pred_id个预测框。
    # gt_id： 第 img_id 张图片上，预测的低gt_id个真值框。
    arg_table = np.zeros((num_pred, 11))  # 用于计算ap的参数表, todo 其实里面有很多冗余信息可以删掉，关键的是conf,hit,p,r,iou

    # iou 阈值范围
    if iou_threshold is None:
        iou_threshold = np.linspace(0.5, 0.95, 10).tolist()  # COCO数据集的评价指标
    else:
        if not isinstance(iou_threshold, list):
            iou_threshold = [iou_threshold]  # 只输入一个iou阈值数值

    ap = []  # 计算不同 iou 下的 AP
    for iou_thr in iou_threshold:
        candidate_idx = 0
        for i, (pred, gt) in enumerate(zip(pred_boxes, gt_boxes)):
            if pred is None:
                continue
            is_match = [False] * len(gt)  # 真值框是否已经有匹配， 每个真值框只能匹配一个预测框

            # 对二维列表某一列为key进行排序：  https://blog.csdn.net/weixin_39544046/article/details/105931060
            pred.sort(key=lambda x: x[4], reverse=True)  # 一张图片上的 预测bbox 根据置信度降序排列

            for pred_id, box1 in enumerate(pred):
                arg_table[candidate_idx, 0] = i  # img_id
                arg_table[candidate_idx, 1] = pred_id
                arg_table[candidate_idx, 3] = box1[4]  # conf

                # 匹配 真值框
                # return a tensor iou of shape = len(gt)
                iou_matrix = bbox_iou(box1[:4], gt, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False)
                top_val, top_idx = torch.topk(iou_matrix, k=len(gt))
                top_val = top_val.tolist()
                top_idx = top_idx.tolist()  # 序号要
                if top_val[0] >= iou_thr and not is_match[top_idx[0]]:
                    arg_table[candidate_idx, 4] = 1  # iou >= thr
                    is_match[top_idx[0]] = True  # 该真值框已经有了预测框
                else:
                    arg_table[candidate_idx, 4] = 0  # iou < thr
                arg_table[candidate_idx, 2] = top_idx[0]
                arg_table[candidate_idx, 10] = top_val[0]

                candidate_idx += 1

        arg_table = arg_table.tolist()
        arg_table.sort(key=lambda x: x[3], reverse=True)  # 所有候选框根据置信度降序排列
        arg_table = np.array(arg_table)

        # 计算 acc TP \ acc FP \ FN \ P \ R
        for i in range(num_pred):
            arg_table[i, 5] = arg_table[:i + 1, 4].sum()  # 累计TP
            arg_table[i, 6] = i + 1 - arg_table[i, 5]  # 累计FP
            arg_table[i, 7] = num_object - arg_table[i, 5]  # FN 漏检框
            arg_table[i, 8] = arg_table[i, 5] / (arg_table[i, 5] + arg_table[i, 6])  # Precision
            arg_table[i, 9] = arg_table[i, 5] / (arg_table[i, 5] + arg_table[i, 7])  # Recall

        # 计算 AP
        r_old = 0
        area_pr = 0
        p_r = arg_table[:, 8:10]
        for p, r in p_r:
            if r == r_old:
                continue
            area_pr += p * (r - r_old)
            r_old = r
        ap.append(area_pr)
    if verbose:  # 打印一些信息
        print(f"arg table（{iou_thr=}）".center(130, "-"))
        print("\timg_id(0)\tpred_id(1)\t\tgt_id(2)\tconf(3)\t\thit(4)"
              "\t\taccTP(5)\taccFP(6)\tNP(7)\tPrecision(8)\tRecall(9)\tiou(10)")
        print(f"{arg_table}\n{ap=}\n{np.mean(ap)=}")

    AP50, AP = ap[0], np.mean(ap)   # 这个地方只有iou_threshold=None才有意义，且AP50, AP75, AP95等都会被计算
    return AP50, AP

