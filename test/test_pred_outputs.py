import torch
import cv2


# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
print(sys.path)

from utils.evaluation import evaluate_pck, evaluate_ap, get_coordinates_from_heatmap
from utils.training_kits import stdout_to_tqdm, load_pretrained_state

from models.pose_hg_ms_att import MultiScaleAttentionHourglass as Network
from data import get_dataset
from config import DATASET, config_dict
from utils.visualization_tools import draw_heatmaps, draw_region_maps, draw_point, draw_bbox, draw_text


class TestPreds:
    def __init__(self, checkpoint="", is_cuda=True, ground_truth=False):

        print("preparing data...")
        self.dataset, self.test_loader = get_dataset(is_train='test')
        print("done!")

        if is_cuda:
            self.device = torch.device(0)
        else:
            self.device = torch.device('cpu')
        self.model = Network().to(self.device)

        self.ground_truth = ground_truth
        if checkpoint != "":
            print("loading state dict...")
            save_dict = torch.load(checkpoint, map_location=self.device)
            print(f"{save_dict.keys()=}")
            
            state, is_match = load_pretrained_state(self.model.state_dict(),
                                                    save_dict['state_dict'])
            # self.model.load_state_dict(save_dict["state_dict"])
            self.model.load_state_dict(state)
            print(f"done! {is_match=}")

    def test(self, n_img=10, show_hms=True, show_kpts=True):
        self.model.eval()
        with torch.no_grad():
            pck = 0
            cv2.namedWindow("img_window")
            if n_img == -1:
                n_img = len(self.test_loader)  # -1 整个数据集过一遍
            for i, (img, target, label, bbox) in enumerate(self.test_loader):
                if i > n_img:
                    break
                print(f"{bbox=}")
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

                pck += evaluate_pck(hm_kpts, target[:, 4:], bbox, thr=0.2).item()
                print(f"{pck=}")

                # 画出热图
                if show_hms:
                    draw_region_maps(region)
                    draw_heatmaps(mask)
                    draw_heatmaps(hm_kpts)

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
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        image = draw_point(img=image, keypoints=kpts)
                        image = self.draw_region_bbox(image, xywh)

                    end = cv2.getTickCount()  # 计时器终点
                    fps = round(cv2.getTickFrequency() / (end - start))
                    print(f"{fps=}")
                    text = str(fps) + "fps"
                    image = draw_text(image, text, (15, 15, 20, 20))
                    cv2.imshow("img_window", image)
                    cv2.waitKey(0)

                    print("*"*200)
                    print("")
            pck = pck / (n_img+1)
            print(f"{n_img=}")
            cv2.destroyAllWindows()
            print(f"{pck=}")

    @staticmethod
    def draw_region_bbox(img, xywhc):
        cx, cy, w, h = xywhc[:4]
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(x2, new_size))
        y2 = int(min(y2, new_size))
        img = draw_bbox(img, x1, y1, x2, y2)
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


if __name__ == '__main__':
    new_size = config_dict["image_size"][0]
    # path = "../record/handnet3_2/2021-09-13/0.716_mPCK_647epoch.pt"
    # path = "../record/handnet3_3/2021-09-13/0.669_mPCK_173epoch.pt"
    # path = "../record/handnet3_1/2021-09-20/0.8_mPCK_86epoch.pt"
    path = "./checkpoint/MSRB-D-DW-PELEE/1HG-ME-att-c256/2021-12-27/0.981_PCK_47epoch.pt"
    # path=""
    t = TestPreds(checkpoint=path, is_cuda=True, ground_truth=False)
    t.test(n_img=-1, show_hms=False, show_kpts=True)


