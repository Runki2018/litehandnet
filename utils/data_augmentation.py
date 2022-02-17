import cv2
import numpy as np
import matplotlib.pyplot as plt
# import pylab as pl
from skimage import exposure
from data.handset.dataset_function import get_bbox
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
        if p > 0.75:
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


def homography(img, keypoints, prob=0.5):
    """
    include the scale, rotate and perspective, so the position of keypoints will be changed
    :param img: cv2 image, (h,w,c)
    :param keypoints: the coordinates of keypoints ->
            [[[x1,y1,1], ...,[x21,y1,1]],
             [[x1,y1,1], ...,[x21,y1,1]],
             ...], (batch, 21,3)
    :param prob: the probability of homography
    """

    if np.random.rand() < prob:
        return img, keypoints

    h, w, _ = img.shape
    src = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    r_horizontal = np.random.randint(10, 15) / 10  # 0.8~1.4
    r_vertical = np.random.randint(10, 15) / 10  # 0.8~1.4
    dx = w * (1 - 1 / r_horizontal) / 2
    dy = h * (1 - 1 / r_vertical) / 2

    x1, y1, x2, y2 = dx, dy, w - 1 - dx, h - 1 - dy  # the left_top and right_bottom of image
    dst = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])

    angle = np.random.randint(-45, 45)
    h_rotate = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1.2)
    pts_src_3 = np.ones((4, 3))
    pts_src_3[:, :2] = dst
    dst = pts_src_3.dot(h_rotate.T)

    h, status = cv2.findHomography(src, dst)  # h is the transformation matrix
    # TODO: 这个填充颜色为了和卷积时的padding保持一致，所以从（128，128，128）改成（0，0，0）
    img_out = cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]), borderValue=(0,0,0))

    # recount the keypoints:
    keypoints_out = keypoints.dot(h.T)

    return img_out, keypoints_out


def horizontal_flip(img, keypoints, prob=0.5):
    """
    :param img: cv2 image, (h,w,c)
    :param keypoints: the coordinates of keypoints ->  (batch, 21 3)
    :param prob: the probability of flip
    """
    if np.random.rand() < prob:
        # img = img[:, ::-1, :]  # img flip: this way is time-consuming
        img = cv2.flip(img, 1)  # img flip
        _, w, _ = img.shape
        keypoints[..., 0] = w - 1 - keypoints[..., 0]  # (batch, 21, 3),[x,y,1], keypoints flip
    return img, keypoints


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

def central_crop(img, bbox, keypoints, size=(256, 256), prob=0.5):
    """
        central crop the image and then scale to specific resolution
    :param size: final resolution after this process， （w, h）
    :param img: cv2 image, (h,w,c)
    :param bbox: (np.array([[cx, cy, w, h]], dtype=np.int16))
    :param keypoints: the coordinates of keypoints ->  (n_hand, 21,3)
    :param prob: the probability of flip
    """
    is_crop = False
    if np.random.rand() < prob:
        return img, bbox, keypoints, is_crop
    
    is_crop = True
    h_img, w_img, c_img = img.shape
    num_object = bbox.shape[0]
    img_crop_list = []

    for i in range(num_object):
        x, y, w, h = bbox[i]
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        w, h = x2 - x1, y2 - y1
        
        img_crop = img[y1:y2, x1:y1]
        factor = min(size[0] / w, size[1] / h)
        nw, nh = w * factor, h * factor
        img_crop = cv2.resize(img_crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # 裁剪图下的关键点坐标
        keypoints[i] -= (x1, y1, 0)
        keypoints[i] *= (factor, factor, 1)

        # 将等长宽比放大后的图像填补为指定宽高的图像
        img_crop = np.pad(img_crop, ((0, size[1] - nh), (0, size[0] - nw), (0, 0)), 'constant', constant_values=128)
        img_crop_list.append(img_crop)
    
    img_crops = np.stack(img_crop_list, axis=0)
    bbox_crop = get_bbox(keypoints, alpha=1.3)
    return img_crops, bbox_crop, keypoints, is_crop
     
    
if __name__ == '__main__':
    img_path = "../test/test_example/1.jpg"
    image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"{image.shape=}")

    pts = np.ones((21, 3), np.int)
    pts[:, :2] *= 400
    # pts[0, :2] = [1, 1]
    # pts[1, :2] = [799, 1023]
    pts[2, :2] = [400, 500]
    pts[3, :2] = [256, 600]
    # pts.resize((1, 21, 3))

    image = adjust_gamma(img=image)
    image = adjust_sigmoid(image)
    image, pts = homography(image, pts)
    image, pts = horizontal_flip(image, pts)
    image, pts = central_scale(image, pts)
    print(f"{pts=}")
    pts.resize((1, 21, 3))
    image = draw_point(image, pts[0])
    cv2.imshow("window", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
