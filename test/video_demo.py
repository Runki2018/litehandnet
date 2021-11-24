""" 打开摄像头，显示录像，并输出保存到本地 """
import cv2
import torch

import os
import sys

sys.path.insert(0, os.path.abspath('./'))
from utils.visualization_tools import *
# from models.handNet3 import HandNet3 as Network
from utils.SPheatmapParser import HeatmapParser_SH
from models.RKNet_SPP import RegionKeypointNetwork as Network
from utils.training_kits import load_pretrained_state
from config.config import  config_dict as cfg
import argparse
from torchvision import transforms


classes_names = ["0-other", "1-okay", "2-palm", "3-up", "4-down",
                 "5-right", "6-left", "7-finger_heart", "8-hush"]
classes_colors = [(0, 0, 255), (147, 20, 255), (255, 0, 0),
                  (0, 255, 255), (0, 255, 255), (0, 70, 255),
                  (208, 224, 64), (130, 0, 75), (193, 182, 255)]

input_size = (512, 512)
transformer = transforms.Compose([
            # transforms.ToTensor(),  # 改变维度含义： （h,w,c） -> (c, h, w)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_coordinates(batch_kpts_hm, img_size=(800, 1280)):
    """ 从关键点热图中获取关键点坐标。"""
    # (batch, n_joints, h, w)
    batch, n_joints, h, w = batch_kpts_hm.shape

    top_val, top_idx = torch.topk(batch_kpts_hm.reshape((batch, n_joints, -1)), k=1)

    batch_kpts = torch.zeros((batch, n_joints, 3))
    batch_kpts[..., 0] = (top_idx % w).reshape((batch, n_joints))  # x
    batch_kpts[..., 1] = (top_idx // w).reshape((batch, n_joints))  # y
    batch_kpts[..., 2] = top_val.reshape((batch, n_joints))  # c: score

    # print(f"{batch_kpts=}")
    # print(f"{w=} \t {h=}")
    # print(f"{torch.tensor(img_size)=}")
    # print(f"{torch.tensor(input_size)=}")
    batch_kpts[..., :2] = batch_kpts[..., :2] * torch.tensor(img_size) / torch.tensor([w, h])
    # batch_kpts[..., :2] = batch_kpts[..., :2] * torch.tensor(input_size) / torch.tensor([w, h])
    # print(f"{batch_kpts=}")
    return batch_kpts


def img_transform(img, size=(512, 512)):
    """将cv2读入的图像处理后，输入网络"""
    size = input_size
    image = cv2.resize(img, size, cv2.INTER_AREA)
    # image = cv2.resize(img, size, cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image) / 255.0
    # mean = image.mean()
    # std = image.std()
    # image = (image - mean) / std
    image = image.permute(2, 0, 1)
    image = transformer(image)
    image = image.unsqueeze(dim=0)
    # print(f"{image.shape=}")

    return image


def draw_multi_bbox(pred_bboxes: list, img, img_size, n=3):
    for bboxes_img in pred_bboxes:
        if bboxes_img is None:
            continue
        for i, bbox in enumerate(bboxes_img):
            if i > n-1:
                break
            print(f"{bbox=}")
            x, y, w, h, conf = bbox
            w_img, h_img = img_size
            x1, y1, x2, y2 = max(0, x-w/2), max(0, y-h/2), min(w_img, x+w/2), min(h_img, y+h/2)
            img = draw_bbox(img, x1, y1, x2, y2)
    return img


def main(options):
    # preparing models
    if options.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
    model = Network()
    save_dict = torch.load(cfg["checkpoint"])
    state, is_match = load_pretrained_state(model.state_dict(), save_dict['model_state'])
    model.load_state_dict(state)

    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    print("models are ready!")

    # heatmap parser
    hm_parser = HeatmapParser_SH()

    # calling camera and test image captured by camera video
    if options.type == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 创建一个VideoCapture对象，此处插上摄像头，参数设置为0
    else:
        cap = cv2.VideoCapture(options.path)  # 创建一个VideoCapture对象，此处插上摄像头，参数设置为0

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取推荐的帧率
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f'fps = {fps}, size = {size}')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID 是一种开源视频编码/解释器，支持avi,mkv,mp4
    out = cv2.VideoWriter('./test/video/out_video1.mp4', fourcc, fps, size)  # 原来fps = 20.0

    cv2.namedWindow("image_window")
    print(" please press 'q' to exit!")
    try:
        while True:
            ret, frame = cap.read()  # 第一个参数返回一个布尔值（True/False），代表有没有读取到图片；第二个参数表示截取到一帧的图片
            if ret:
                start = cv2.getTickCount()  # 计时器起点,用于计算帧率

                # frame = cv2.imread("./test/test_example/2_256x256.jpg")
                # size = (256, 256)

                image = img_transform(frame, input_size)  # 处理图像为模型指定输入格式
                # predict bbox and keypoints by simpleHRNet
                pred = model(image.to(device))  # shape = (1,21,3) = (nof_person/images, nof_joints, xyc/yxc?)
                heatmaps = pred[2][:, 1:]
                center_maps = pred[2][:, 0:1]
                size_maps = pred[4]
                kpt, pred_bboxes = hm_parser.parse(heatmaps, center_maps, size_maps, size)
                kpt = kpt.squeeze(dim=0).numpy()
                # kpt[:, :, 0] = kpt[:, :, 0].clip(0, size[0]-1)
                # kpt[:, :, 1] = kpt[:, :, 1].clip(0, size[1]-1)
                # kpt = get_coordinates(pred[2], size)
                # kpt = kpt.squeeze(dim=0)[1:].detach().numpy()

                # frame = cv.cv2.resize(frame, input_size, cv2.INTER_AREA)
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = draw_point(frame, kpt)
                frame = draw_multi_bbox(pred_bboxes, frame, size)

                end = cv2.getTickCount()  # 计时器终点
                fps = round(cv2.getTickFrequency() / (end - start))
                text = str(fps) + " fps"
                frame = draw_text(frame, text, [15, 15, 40, 40], color=classes_colors[0])

                cv2.imshow('image_window', frame)
                # out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="输入相应命令来执行摄像头测试或已有视频的测试")
    parser.add_argument('--type', type=int, default=1, help="0是摄像头测试，1是本地视频测试")
    parser.add_argument('--path', type=str, default="./video/hand_video001.avi", help="本地视频的路径")
    parser.add_argument('--cpu', action='store_true', default=False, help="是否用CPU运行，默认为假")
    args = parser.parse_args()
    # print(args.cpu)
    main(args)
