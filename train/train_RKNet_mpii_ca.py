import os
import time
import torch
from torch import optim
import numpy as np

from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter

from utils.evaluation import evaluate_pck, evaluate_ap
from utils.top_down_eval import get_PCK
from utils.training_kits import stdout_to_tqdm, load_pretrained_state
from utils.SPheatmapParser import HeatmapParser_SH

# from models.RKNet_SPP import RegionKeypointNetwork_SPP as Network
# from models.RKNet_mask_ca import RegionKeypointNetwork as Network
# from models.RKNet_Attention import RKNet_attention as Network
# from models.RKNet_parallel import RegionKeypointNetwork as Network
from models.RKNet_Bulat import RegionKeypointNetwork_Bulat as Network
from loss.loss import HMLoss as Criterion
from data import get_dataset
from config.config import DATASET, config_dict as cfg

os.environ['CUDA_VISIBLE_DEVICES'] = cfg["CUDA_VISIBLE_DEVICES"]
device_id = int(cfg["device_id"])
exp_id = cfg["experiment_id"]

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Main:
    """
        训练 HandNet
    """

    def __init__(self):
        # 定义参数
        self.epoch = 0  # 当前训练的epoch
        self.start_epoch = 0  # 开始的epoch，如果是重新开始训练为0，如果是继续训练则接着上次训练的epoch开始
        self.end_epoch = cfg["n_epochs"]  # 训练结束的epoch
        self.num_hourglass = cfg["num_hourglass"]
        self.batch = cfg['batch_size']
        self.hm_size = cfg['hm_size']
        self.test_interval = cfg['test_interval']

        self.save_root = cfg["save_root"] + exp_id + "/"

        # 定义模型
        self.model = Network()
        # todo 实现多GPU训练, 注意保持模型参数会有所区别
        # available_cuda = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        # print(f"{available_cuda=}")
        self.model.cuda(device_id)
        print("Model is ready!")

        # 定义损失和优化器 todo 这部分都可以在确认模型没有大问题后，多次实验不同的优化器和学习率以寻找最佳参数
        self.criterion = Criterion()
        # self.criterion = SmoothL1()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=cfg["lr"], momentum=0.4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg["lr"])
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=cfg["lr"])

        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg["step_size"], gamma=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=20,
                                                              min_lr=cfg["lr"] * 0.001, cooldown=10)

        # 数据集和数据加载器
        _, self.trainLoader = get_dataset('train')
        self.test_set, self.testLoader = get_dataset('test')

        self.writer = SummaryWriter(logdir=self.save_root + 'log')  # 记录训练参数日志
        self.parser = HeatmapParser_SH()  # 解析热图结果

        # 加载检测点
        if cfg["reload"]:
            self.save_dict = torch.load(cfg["checkpoint"])
            state, is_match = load_pretrained_state(self.model.state_dict(), self.save_dict['model_state'])
            self.model.load_state_dict(state)
            if not cfg["just_model"] and is_match:  # 有时后只是需要预训练权重，不需要其他超参数
                self.optimizer.load_state_dict(self.save_dict["optimizer"])
                self.start_epoch = self.save_dict["epoch"]
                self.best_loss = min(self.save_dict["loss"])
                self.best_mPCK = max(self.save_dict["mPCK"])

        # 存储参数的字典
        self.best_loss = 0
        self.best_mPCK = 0
        self.best_ap = 0
        self.save_dict = {"epoch": 0, "lr": [], "loss": [], "mPCK": [], "ap": [],
                          "model_state": {}, "optimizer": {}, "config": cfg}

    def train(self):
        self.model.train()
        sum_loss = 0
        with stdout_to_tqdm() as save_stdout:
            for img, target, hm_weight, bbox, meta in tqdm(self.trainLoader, file=save_stdout, ncols=8):

                self.optimizer.zero_grad()
                hm_list = self.model(img.cuda(device_id))

                target = target.cuda(device_id)
                loss, ll = self.criterion(hm_list, target, hm_weight)

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    sum_loss += loss.item()
                    print(exp_id, end="")
                    for i in range(self.num_hourglass + 1):
                        print("{} ->\t kpt: {:.4f}".format(i, ll[i]))
                    print("mask:{}\tregion: {:.4f}".format(ll[-2], ll[-1]))

        print(f"{sum_loss=}")
        lr = self.optimizer.param_groups[0]['lr']
        print(f"{lr=}")
        if lr > 5e-20:
            self.scheduler.step(metrics=sum_loss)

        # 记录训练数据
        self.save_dict["lr"].append(lr)
        self.save_dict["loss"].append(sum_loss)

        best_loss = min(self.save_dict["loss"])
        if sum_loss <= best_loss:
            self.best_loss = sum_loss
            self.save_model(best_loss=True)

        self.writer.add_scalar("sum_loss", sum_loss, self.epoch)
        self.writer.add_scalar("lr", lr, self.epoch)

    def test(self):
        self.model.eval()
        with torch.no_grad():

            mPCK = [0] * (self.num_hourglass + 1)
            mAP50 = []  # iou_thr 0.5
            mAP = []  # iou_thr 0.5~0.95
            region_maps = torch.zeros((self.batch, 3, self.hm_size[1], self.hm_size[0]))
            for img, target, hm_weight, bbox, meta in tqdm(self.testLoader, desc="testing "):
                hm_list = self.model(img.cuda(device_id))
                for i in range(len(mPCK)):
                    mPCK[i] += evaluate_pck(hm_list[i][:, 1:], target[:, 4:],
                                            bbox, target_weight=hm_weight[:, 1:]).item() * target.shape[0]

                region_maps[:, 0] = hm_list[self.num_hourglass][:, 0]
                region_maps[:, 1:] = hm_list[-1]
                # ap50, ap,  _ = evaluate_ap(region_maps, bbox)
                kpt, pred_bboxes = self.parser.parse(hm_list[self.num_hourglass][:, 1:], region_maps[:, 0:1],
                                                     region_maps[:, 1:], cfg['image_size'])

                ap50, ap = self.parser.evaluate_ap(pred_bboxes, bbox)
                mAP50.append(ap50 * self.batch)
                mAP.append(ap * self.batch)

            mPCK = [pck / self.testLoader.__len__() / self.batch for pck in mPCK]
            mAP50 = sum(mAP50) / self.testLoader.__len__() / self.batch
            mAP = sum(mAP) / self.testLoader.__len__() / self.batch
            for i, pck in enumerate(mPCK):
                print(f"{i} --> { pck=}")

            print(f"{mAP50=} \t {mAP=}")
            # 记录训练数据
            self.save_dict["mPCK"].append(mPCK[-1])
            self.save_dict["ap"].append(mAP)

            self.writer.add_scalars('mPCK', {f"pck_{i}": pck for i, pck in enumerate(mPCK)}, self.epoch)
            self.writer.add_scalars('AP50/AP', {'ap50': mAP50, 'ap': mAP}, self.epoch)

            if len(self.save_dict["mPCK"]) > 3:
                for i in range(-1, -4, -1):
                    print("pck{} = {:.5f}".format(i, self.save_dict["mPCK"][i]), end="\t")
            else:
                print("pck = {:.5f}".format(mPCK[-1]))

            if mPCK[-1] > self.best_mPCK:
                self.best_mPCK = mPCK[-1]
                self.save_model(best_pck=True)

            if mAP > self.best_ap:
                self.best_ap = mAP
                self.save_model(best_ap=True)

    def save_model(self, best_loss=False, best_pck=False, best_ap=False):
        save_dir = self.save_root + time.strftime("%Y-%m-%d", time.localtime())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if best_loss:
            file_name = "/{0}_loss_{1}epoch.pt".format(round(self.best_loss, 3), self.epoch)
        elif best_pck:
            file_name = "/{0}_mPCK_{1}epoch.pt".format(round(self.best_mPCK, 3), self.epoch)
        elif best_ap:
            file_name = "/{0}_ap_{1}epoch.pt".format(round(self.best_ap, 3), self.epoch)
        else:
            raise NotImplementedError
        save_file = save_dir + file_name
        print(f"{save_file=}")

        self.save_dict["epoch"] = self.epoch
        self.save_dict["model_state"] = self.model.state_dict()
        self.save_dict["optimizer"] = self.optimizer.state_dict()
        torch.save(self.save_dict, save_file)

    def run(self):
        try:
            for epoch in range(self.start_epoch, self.end_epoch):
                print('\nExp %s Epoch %d of %d @ %s' %
                      (exp_id, epoch + 1, self.end_epoch, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                self.epoch = epoch

                self.train()
                if (epoch % self.test_interval == 0) or (epoch == self.end_epoch -1) :
                    self.test()
        finally:
            self.writer.close()


if __name__ == '__main__':
    f = Main()
    f.run()
