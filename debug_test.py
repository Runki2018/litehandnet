import torch
import numpy as np
import cv2
from utils.training_kits import stdout_to_tqdm, load_pretrained_state

from data import get_dataset
from config import DATASET, config_dict
from utils.visualization_tools import draw_heatmaps, draw_region_maps, draw_point, draw_bbox, draw_text,draw_centermap
from models.center_simDR import LiteHourglassNet as Network
from utils.CenterSimDRParser import ResultParser

from tensorboardX import SummaryWriter
import torchvision.utils as vutils

config_dict['batch_size'] = 1
config_dict['workers'] = 1

new_size = config_dict["image_size"][0]
def draw_region_bbox(img, xywhc):
    cx, cy, w, h = xywhc[:4]
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(x2, new_size))
    y2 = int(min(y2, new_size))
    img = draw_bbox(img, x1, y1, x2, y2)
    return img

#  1、用于可视化用SimDR检测关键点的手部检测
img, target_x, target_y, target_weight, centermap, centermask, bbox, gt_kpts = \
None, None, None, None, None, None, None, None
class TestPreds:
    def __init__(self, checkpoint="", is_cuda=True, ground_truth=False, exp_name=''):

        print("preparing data...")
        self.dataset, self.test_loader = get_dataset(set_type='test')
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
        
        self.writer = SummaryWriter(log_dir='jupyter_log/'+ exp_name)
        self.parser = ResultParser()

    def test(self, n_img=10, show_hms=True, show_kpts=True):
        self.model.eval()
        with torch.no_grad():
            pck = 0
            if n_img == -1:
                n_img = len(self.test_loader)  # -1 整个数据集过一遍
            for i, meta in enumerate(self.test_loader):
                if i > n_img:
                    break
                start = cv2.getTickCount()  # 计时器起点

                img, target_x, target_y, target_weight, centermap, centermask, bbox, gt_kpts = meta
                print(f"{bbox=}")
                print(f"{target_x[target_x > 0.5]=}")
                print(f"{target_y[target_y > 0.5]=}")
                print(f"{centermap=}")

                
                if not self.ground_truth:
                    pred_centermap, pred_x, pred_y = self.model(img.to(self.device))
                else:
                    pred_centermap, pred_x, pred_y = centermap, target_x, target_y
         
                # 结果解析，得到原图关键点和边界框
                pred_kpts, pred_bboxes = self.parser.parse(pred_centermap, pred_x, pred_y)
                ap50, ap = self.parser.evaluate_ap(pred_bboxes, bbox)

                # 画出热图
                if show_hms:
                    out_centermap = draw_centermap(centermap)
                    out_centermap = [torch.tensor(out, dtype=torch.uint8) for out in out_centermap]
                    imgs = torch.stack(out_centermap, dim=0)
                    self.writer.add_image('centermap', imgs, i, dataformats='NHWC')

                # 画出关键点
                if show_kpts:
                    batch_xywh = pred_bboxes[0]
                    if batch_xywh is None:
                        print("没有找到目标框")
                        batch_xywh = [[4, 4, 2, 2, 0]]

                    for image, kpts, xywh in zip(img, pred_kpts, batch_xywh):  
                    # for image, kpts, xywh in zip(img, gt_kpts, batch_xywh):    
                        image = image.permute(1, 2, 0).detach().numpy()
                        m = np.array([0.485, 0.456, 0.406])
                        s = np.array([0.229, 0.224, 0.225])
                        image = image * s + m
                        image *= 255
                        image = image.astype(np.uint8) 
                        
                        # kpts = kpts.squeeze(dim=0).detach().cpu().numpy()
                        kpts = kpts[0].detach().cpu().numpy()
                        # print(f"{image.shape=}")
                        print(f"{kpts.shape=}")
                        print(f"{kpts=}")
                        print(f"{gt_kpts[0, 0].shape=}")
                        print(F"{gt_kpts[0, 0]=}")
                        print('*'* 100)
                        
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        image_drawn = draw_point(img=image.copy(), keypoints=kpts)
                        image_drawn = draw_region_bbox(image_drawn, xywh)

                    end = cv2.getTickCount()  # 计时器终点
                    fps = round(cv2.getTickFrequency() / (end - start))
                    text = str(fps) + "fps"
                    img = draw_text(image_drawn, text, (15, 15, 20, 20))
                    # img = img[:,:,::-1]   # BGR to RGB
                    imgs = torch.stack([torch.tensor(image, dtype=torch.uint8),
                                        torch.tensor(image_drawn, dtype=torch.uint8)], dim=0)
                    self.writer.add_images('images', imgs, i, dataformats='NHWC')
                    
            pck = pck / (n_img+1)
            print(f"{n_img=}")
            print(f"{pck=}")
            self.writer.close()
            
# path = "./checkpoint/MSRB-D-DW-PELEE/1HG-ME-att-c256/2021-12-27/0.981_PCK_47epoch.pt"
path = "./checkpoint/Center_SimDR/1HG-lite/2022-01-04/0.046_AP_493epoch.pt"
t = TestPreds(checkpoint=path, is_cuda=False, ground_truth=True, exp_name='cs')
t.test(n_img=20, show_hms=False, show_kpts=True)