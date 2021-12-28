import math
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import torch
import cv2

from utils.evaluation import evaluate_pck, evaluate_ap, get_coordinates_from_heatmap
from utils.top_down_eval import pose_pck_accuracy
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# sys.path.insert(0, os.path.abspath('..' + '/'))
# print(sys.path)

from models.hourglass_SA import HourglassNet_SA as Network
from data import get_dataset
from config.config import DATASET, config_dict as cfg
from utils.training_kits import load_pretrained_state
from utils.visualization_tools import draw_heatmaps, draw_region_maps, draw_point, draw_bbox, draw_text

from utils.transforms import transform_preds, get_affine_transform


class TestPreds:
    def __init__(self, checkpoint="", is_cuda=True, ground_truth=False):

        print("preparing data...")
        self.dataset, self.test_loader = get_dataset(set_type='test')
        print("done!")

        self.batch_size = cfg['batch_size']
        if is_cuda:
            self.device = torch.device(0)
        else:
            self.device = torch.device('cpu')
        self.model = Network().to(self.device)

        self.ground_truth = ground_truth
        if checkpoint != "":
            print("loading state dict...")
            save_dict = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(save_dict["state_dict"])
            # state, is_match = load_pretrained_state(self.model.state_dict(), save_dict['model_state'])
            # self.model.load_state_dict(state)
            print("done!")

    def test(self, n_img=10, show_hms=True, show_kpts=True):
        try:
            self.model.eval()
            with torch.no_grad():
                pck = 0
                cv2.namedWindow("img_window")
                if n_img == -1:
                    n_img = len(self.test_loader)  # -1 整个数据集过一遍

                pred_kpts = []
                for i, (img, target, hm_weight, bbox, meta) in enumerate(self.test_loader):
                    if i > n_img:
                        break
                    # print(f"{bbox=}")
                    print(f"{bbox.shape=}")

                    start = cv2.getTickCount()  # 计时器起点

                    region = torch.zeros_like(target[:, :3])
                    region[:, 0:1] = target[:, 3:4]
                    region[:, 1:] = target[:, :2]
                    mask = target[:, 2:3]
                    hm_kpts = target[:, 4:].to(self.device)

                    if not self.ground_truth:
                        hms_list = self.model(img.to(self.device))  # [22, 22, 22, 44, 88]
                        hm_kpts = hms_list[2][:, 1:]
                        region = torch.zeros_like(target[:, :3])
                        region[:, 0:1] = hms_list[2][:, 0:1]
                        region[:, 1:] = hms_list[-1]
                        mask = hms_list[3]

                    # pck += evaluate_pck(hm_kpts, target[:, 4:], bbox, thr=0.2).item()
                    # hm_kpts = hm_kpts.numpy()
                    # target = target[:, 4:].numpy()
                    # vis = hm_weight[:, 1:, 0].numpy()
                    # # print(f"{vis=}")
                    # print(f"{hm_kpts.shape=}\t{target.shape=}\t{vis.shape=}")
                    # acc, avg_acc, _ = pose_pck_accuracy(hm_kpts, target, vis)
                    # print(f"{acc=}\t{avg_acc=}")
                    # pck += avg_acc * self.batch_size

                    c = meta['center'].numpy()
                    s = meta['scale'].numpy()
                    preds = self.get_final_preds(hm_kpts.clone().cpu(), c, s)
                    batch_size = preds.shape[0]
                    for bi in range(batch_size):
                        pred_kpts.append(preds[bi])

                    # 画出关键点
                    if show_kpts:
                        # batch_xywh = cs_from_region_map(batch_region_maps=region, k=1, thr=0.1)
                        _, _, batch_xywh = evaluate_ap(region, bbox, k=10, conf_thr=0.1)
                        print(f"{batch_xywh=}")
                        batch_xywh = batch_xywh[0]
                        if batch_xywh is None:
                            print("没有找到目标框")
                            batch_xywh = [[4, 4, 2, 2, 0]]

                        batch_kpts, _ = get_coordinates_from_heatmap(hm_kpts)
                        # batch_kpts = self.get_coordinates(hm_kpts[:, 4:])
                        print(f"{batch_kpts.shape=}")

                        heatmaps_size = hm_kpts.shape[-1]
                        batch_kpts[..., :2] = batch_kpts[..., :2] * new_size / heatmaps_size  # scale to original size

                        for image, kpts, xywh in zip(img, batch_kpts, batch_xywh):
                            image = image.permute(1, 2, 0).detach().numpy()

                            kpts = kpts.squeeze(dim=0).detach().cpu().numpy()
                            print(f"{image.shape=}")
                            print(f"{kpts.shape=}")
                            print(f"{xywh=}")
                            # xywh[2] *= 1.2
                            # xywh[3] *= 1.2
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            image = draw_point(img=image, keypoints=kpts)
                            image = self.draw_region_bbox(image, xywh)  # pred bbox
                            image = self.draw_region_bbox(image, bbox[0, 0], (0, 255, 0))  # gt bbox

                        end = cv2.getTickCount()  # 计时器终点
                        fps = round(cv2.getTickFrequency() / (end - start))
                        print(f"{fps=}")
                        text = str(fps) + "fps"
                        image = draw_text(image, text, (15, 15, 20, 20))
                        cv2.imshow("img_window", image)
                        cv2.waitKey(0)

                        print("*" * 200)
                        print("")

                    # 画出热图
                    if show_hms:
                        draw_region_maps(region)
                        draw_heatmaps(mask)
                        draw_heatmaps(hm_kpts)

                pck = pck / (n_img * self.batch_size)
                print(f"{n_img=}")
                print(f"total {pck=}")

                preds = np.stack(kpts)
                print(f"{preds.shape=}")
                v, _ = self.dataset.evaluate(preds, output_dir='../record/handnet3_mask_ca2_1')
                print(v)
        finally:
            cv2.destroyAllWindows()

    @staticmethod
    def draw_region_bbox(img, xywhc, color=(0, 0, 255)):
        cx, cy, w, h = xywhc[:4]
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(x2, new_size))
        y2 = int(min(y2, new_size))
        img = draw_bbox(img, x1, y1, x2, y2, color)
        return img

    @staticmethod
    def get_coordinates(batch_kpts_hm):
        # (batch, n_joints, h, w)
        batch, n_joints, h, w = batch_kpts_hm.shape
        top_val, top_idx = torch.topk(batch_kpts_hm.reshape((batch, n_joints, -1)), k=1)

        batch_kpts = torch.zeros((batch, n_joints, 3))
        batch_kpts[..., 0] = (top_idx % w).reshape((batch, n_joints))  # x
        batch_kpts[..., 1] = (top_idx // w).reshape((batch, n_joints))  # y
        batch_kpts[..., 2] = top_val.reshape((batch, n_joints))  # c: score

        return batch_kpts

    def get_final_preds(self, batch_heatmaps, center, scale):
        coords = self.get_coordinates(batch_heatmaps)

        h_heatmap = batch_heatmaps.shape[2]
        w_heatmap = batch_heatmaps.shape[3]

        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < w_heatmap - 1 and 1 < py < h_heatmap - 1:
                    diff = np.array([
                        hm[py][px + 1] - hm[py][px - 1],
                        hm[py + 1][px] - hm[py - 1][px]
                    ])
                    coords[n][p][..., :2] += np.sign(diff) * .25
        preds = coords.numpy()

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(
                coords[i], center[i], scale[i], [w_heatmap, h_heatmap]
            )

        return preds


if __name__ == '__main__':
    new_size = cfg["image_size"][0]
    # path = "checkpoint/2HG_1/2021-12-08/87.554_mPCK_21epoch.pt"
    path = ""

    t = TestPreds(checkpoint=path, is_cuda=True, ground_truth=True)
    t.test(n_img=-1, show_hms=True, show_kpts=True)
