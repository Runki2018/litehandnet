import cv2
import numpy as np
import matplotlib.pyplot as plt
# import pylab as pl
from skimage import exposure
# from data.handset.dataset_function import get_bbox
from utils.visualization_tools import draw_point

# https://zhuanlan.zhihu.com/p/133707658
# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def adjust_gamma(img, prob=0.5):
    """
    gamma transformation is conducive to images which is too dark or too light
     just adjust the value of pixels, not affair the position of keypoints
    :param prob: probability of performing transformation
    :param img: cv2 image
    :return: an image processed by gamma transformation randomly
    """

    p = np.random.rand()
    if p < prob:
        if p < prob / 2:
            gamma = np.random.randint(2, 10) / 10  # 0.2~1, 50% to brighten the image
        else:
            gamma = np.random.randint(1, 3)  # 1~3, 50% to darken the image
        img = exposure.adjust_gamma(image=img, gamma=gamma)
    return img


def adjust_sigmoid(img, prob=0.5):
    """Performs Sigmoid Correction on the input image
        just adjust the value of pixels, not affair the position of keypoints
    """
    p = np.random.rand()
    if p < prob:
        gain = np.random.randint(3, 5)  # 3 or 4
        img = exposure.adjust_sigmoid(image=img, gain=gain)
    return img


def homography(img, keypoints, prob=0.5, bbox=None, topleft=False):
    """
    include the scale, rotate and perspective, so the position of keypoints will be changed
    :param img: cv2 image, (h,w,c)
    :param keypoints: the coordinates of keypoints ->
            [[[x1,y1,1], ...,[x21,y1,1]],
             [[x1,y1,1], ...,[x21,y1,1]],
             ...], (n_hand, 21,3)
    :param prob: the probability of homography
    """

    if np.random.rand() < prob:
        if bbox is None:
            return img, keypoints
        else:
            return img, keypoints, bbox

    H, W, _ = img.shape
    src = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]])
    r_horizontal = np.random.randint(10, 14) / 10 
    r_vertical = np.random.randint(10, 14) / 10  
    dx = W * (1 - 1 / r_horizontal) / 2
    dy = H * (1 - 1 / r_vertical) / 2

    x1, y1, x2, y2 = dx, dy, W - 1 - dx, H - 1 - dy  # the left_top and right_bottom of image
    dst = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])

    angle = np.random.randint(-30, 30)
    h_rotate = cv2.getRotationMatrix2D((H / 2, W / 2), angle, 1.2)
    pts_src_3 = np.ones((4, 3))
    pts_src_3[:, :2] = dst
    dst = pts_src_3.dot(h_rotate.T)

    mat, status = cv2.findHomography(src, dst)  # h is the transformation matrix
    # TODO: 这个填充颜色为了和卷积时的padding保持一致，所以从（128，128，128）改成（0，0，0）
    img_out = cv2.warpPerspective(img, mat, (W, H), borderValue=(0,0,0))

    # recount the keypoints:
    keypoints_out = keypoints.dot(mat.T)

    if bbox is None:
        return img_out, keypoints_out
    else:
        bbox = _affine_bbox(bbox, mat, topleft, (W, H))
        
        return img_out, keypoints_out, bbox

def _affine_bbox(bbox, mat, topleft, size):
    """目标检测中,是不进行旋转变换的,即仿射变换时,最大旋转角度设置为零,因为旋转后的GT-BBOX,与关键点不同,旋转后的GT-BBOX可能不能很好的贴合目标边界, 所以affine_bbox这个自行设计的函数其实没有必要,也不合理。
    """
    bbox = np.array(bbox)  # [num_people, 4]
    num_people = len(bbox)
    if topleft:  # lx ly w, h
        x1y1 = bbox[:, :2]
    else: # cx, cy, w, h
        x1y1 = bbox[:, :2] - bbox[:, 2:] / 2
    x2y2 = x1y1 + bbox[:, 2:]
    x1y2 = np.concatenate([x1y1[:, 0:1], x2y2[:, 1:2]], axis=1)
    x2y1 = np.concatenate([x2y2[:, 0:1], x1y1[:, 1:2]], axis=1)
        
    vertex = np.concatenate([x1y1, x1y2, x2y1, x2y2], axis=0)
    vertex = np.concatenate([vertex, vertex[:, 0:1]*0+1], axis=1)   # 3 tuple vector
    vertex = np.dot(vertex, mat.T)
    
    vertex = vertex[:, :2].reshape((num_people, 4, 2))
    tl = np.min(vertex, axis=1)  # top-left-vertex
    br = np.max(vertex, axis=1)  # bottom-right-vertex
    
    W, H = size
    tl = np.maximum(tl, 0)
    br[:, 0] = np.minimum(br[:, 0], W - 1)
    br[:, 1] = np.minimum(br[:, 1], H - 1)
    
    if topleft:
        bbox[:, :2] = tl
    else:
        bbox[:, :2] = (tl + br) / 2
    bbox[:, 2:] = br - tl   
    return bbox

def horizontal_flip(img, keypoints, prob=0.5, bbox=None):
    """
    :param img: cv2 image, (h,w,c)
    :param keypoints: the coordinates of keypoints ->  (batch, 21 3)
    :param prob: the probability of flip
    """
    if np.random.rand() < prob:
        # img = img[:, ::-1, :]  # img flip: this way is time-consuming
        img = cv2.flip(img, 1)  # img flip
        _, w, _ = img.shape
        keypoints[:, :, 0] = w - 1 - keypoints[:, :, 0]  # (batch, 21, 3),[x,y,1], keypoints flip
        if bbox is not None:
            bbox[:, 0] = w - 1 - bbox[:, 0]
    if bbox is None:
        return img, keypoints
    else:
        return img, keypoints, bbox


def central_scale(img, keypoints, resolution=(256, 256), prob=0.5):
    """
        central crop the image and then scale to specific resolution
    :param resolution: final resolution after this process
    :param img: cv2 image, (h,w,c)
    :param keypoints: the coordinates of keypoints ->  (21,3)
    :param prob: the probability of flip
    """
    # todo: just use in the situation of one hand.
    if np.random.rand() < prob:
        return img, keypoints

    x1y1 = keypoints.min(0)[:2]  # array([x1, y1, 1])
    x2y2 = keypoints.max(0)[:2]  # array([x2, y2, 1])

    h, w, _ = img.shape
    wh = np.array([w, h])
    p = np.random.rand(2, 2)  # (2,2)
    x1, y1 = (x1y1 - p[0]*x1y1).astype(np.int32)  # (2)
    x2, y2 = (x2y2 + (wh-x2y2)*p[1]).astype(np.int32)  # (2)
    # img_crop = img[y1:y2, x1:x2]
    src = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    dst = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    H, status = cv2.findHomography(src, dst)  # h is the transformation matrix
    print(f"{H=}")
    img_crop = cv2.warpPerspective(img, H, (w, h), borderValue=(128, 128, 128))
    keypoints_out = keypoints.dot(H.T)

    return img_crop, keypoints_out

