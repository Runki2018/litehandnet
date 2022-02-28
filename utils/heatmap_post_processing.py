import torch
import numpy as np
import cv2
from config.config import pcfg

def adjust_keypoints_by_offset(keypoints, heatmaps):
        """
        :param keypoints: (batch, n_joints, 3), last dim = [x, y , conf]
        :param heatmaps: (batch, n_joints, hm_size, hm_size)  在nms前的关键点预测热图
        :return: (list) keypoints after adjustment
        """
        batch, n_joints, _ = keypoints.shape
        offset = 0.25 
        keypoints = keypoints.detach()
        for batch_id in range(batch):
                for joint_id in range(n_joints):
                    x, y = keypoints[batch_id, joint_id, :2]  # column, row
                    xx, yy = int(x), int(y)
                    # print(f"{xx=}\t{yy=}")
                    tmp = heatmaps[batch_id, joint_id]  # (h, w)
                    if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                        x += offset  # 如果峰值点右侧点 > 左侧点，则最终坐标向右偏移0.25
                    else:
                        x -= offset

                    if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(yy - 1, 0), xx]:
                        y += offset
                    else:
                        y -= offset
                     # todo 在simpleHigherHRNet上这个是 x + 0.5
                    keypoints[batch_id, joint_id, 0] = x + 0.5
                    keypoints[batch_id, joint_id, 1] = y + 0.5
        return keypoints
  
def adjust_keypoints_by_DARK(keypoints, heatmaps):
        """
        :param keypoints: (batch, n_joints, 3), last dim = [x, y , conf]
        :param heatmaps: (batch, n_joints, hm_size, hm_size)  在nms前的关键点预测热图
        :return: (list) keypoints after adjustment
        """
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.detach().cpu().numpy()
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.detach().cpu().numpy()
        
        # post-processing
        hm = gaussian_blur(heatmaps, pcfg['blue_kernel'])   #  k = 11
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for i in range(keypoints.shape[0]):  # index of images
                for k in range(keypoints.shape[1]):  # index of  joints
                    keypoints[i, k] = taylor(hm[i][k], keypoints[i][k])

        return keypoints
        
def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
        dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
        dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
        dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
        dxy = 0.25 * (hm[py + 1][px + 1] - hm[py - 1][px + 1] - hm[py + 1][px - 1] \
                      + hm[py - 1][px - 1])
        dyy = 0.25 * (hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
        derivative = np.matrix([[dx], [dy]])
        hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord[:2] += offset
    return coord

def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2  # 5 = (11 - 1) // 2
    batch_size = hm.shape[0]
    num_joints = hm.shape[1]
    height = hm.shape[2]
    width = hm.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(hm[i, j])  # scalar
            dr = np.zeros((height + 2 * border, width + 2 * border))  # 64 + 2 * 5 = 74
            dr[border: -border, border: -border] = hm[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            hm[i, j] = dr[border: -border, border: -border].copy()
            hm[i, j] *= origin_max / (np.max(hm[i, j]) + 1e-6)
    return hm   
 
        
        