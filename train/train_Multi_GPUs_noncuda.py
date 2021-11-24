import os
import time
import torch
from torch import optim
import numpy as np

import time
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter

from utils.evaluation import evaluate_pck
from utils.SPheatmapParser import HeatmapParser_SH
from utils.training_kits import stdout_to_tqdm, load_pretrained_state
from models.ResultAttention import RANet as Network
from loss.loss import HMLoss as Loss
from data import get_dataset
from config.config import config_dict as cfg

os.environ['CUDA_VISIBLE_DEVICES'] = cfg["CUDA_VISIBLE_DEVICES"]
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

        self.save_root = cfg["save_root"] + exp_id + "/"

        # 数据集和数据加载器
        _, self.trainLoader = get_dataset('train')
        self.test_set, self.testLoader = get_dataset('test')

        self.writer = SummaryWriter(logdir=self.save_root + 'log')  # 记录训练参数日志
        self.hm_parser = HeatmapParser_SH()  # 解析热图预测结果

        # 多GPU训练
        available_cuda = [i for i in range(torch.cuda.device_count())]
        print(f"{available_cuda=}")
        self.model = torch.nn.DataParallel(Network(), device_ids=available_cuda)
        print("Model is ready!")

        # 定义损失和优化器 todo 这部分都可以在确认模型没有大问题后，多次实验不同的优化器和学习率以寻找最佳参数
        # self.optimizer = optim.SGD(self.model.parameters(), lr=cfg["lr"], momentum=0.4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg["lr"])
        # self.optimizer = optim.RMSprop(self.model.module.parameters(), lr=cfg["lr"])

        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg["step_size"], gamma=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=20,
                                                              min_lr=cfg["lr"] * 0.001, cooldown=10)

        self.criterion = Loss()
        for state in self.optimizer.state.values():
            for k, v in state.items():  # optimizer加载参数时,tensor默认在CPU上
                if torch.is_tensor(v):
                    state[k] = v

        # 加载检测点
        if cfg["reload"]:
            self.save_dict = torch.load(cfg["checkpoint"])
            if 'model_state' in self.save_dict:
                state, is_match = load_pretrained_state(self.model.module.state_dict(), self.save_dict['model_state'])
            elif 'state_dict' in self.save_dict:
                state, is_match = load_pretrained_state(self.model.module.state_dict(), self.save_dict['state_dict'])
            else:
                state, is_match = load_pretrained_state(self.model.module.state_dict(), self.save_dict)
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
            for img, target, target_weight, _ in tqdm(self.trainLoader, file=save_stdout, ncols=8):
                self.optimizer.zero_grad()

                output = self.model(img)
                loss = self.criterion(output, target, target_weight)

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    sum_loss += loss.item()

        print(exp_id, f"{sum_loss=}")
        lr = self.optimizer.param_groups[0]['lr']
        print(f"{lr=}")
        if lr > 5e-7:
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
            n_out = cfg['nstack'] + 1
            batch_size = cfg['batch_size']
            pck = [0] * n_out
            pred_kpts = []

            for img, target, target_weight, meta in tqdm(self.testLoader, desc="testing "):
                hm_list = self.model(img)
                for i in range(n_out):
                    if len(pred_kpts) == i:
                        pred_kpts.append([])
                    kpt, _ = self.hm_parser.parse(hm_list[i], image_size=cfg['image_size'])  # [B, K, 3]
                    pred_kpts[i].append(kpt)

            for i in range(n_out):
                preds = np.concatenate(pred_kpts[i])  # [n_images, n_joints, 3]

                name_value, PCKh = self.test_set.evaluate(preds, cfg['out_dir'])
                print(f"PCKh_{i}\n {name_value}")
                pck[i] = PCKh

            # 记录训练数据
            self.save_dict["mPCK"].append(pck[-1])
            self.writer.add_scalars('mPCK', {f'PCKh_{i}': p for i, p in enumerate(pck)}, self.epoch)

            if len(self.save_dict["mPCK"]) > 3:
                for i in range(-1, -4, -1):
                    print("pck{} = {:.5f}".format(i, self.save_dict["mPCK"][i]), end="\t")
            else:
                print("pck = {:.5f}".format(pck[-1]))

            if pck[-1] > self.best_mPCK:
                self.best_mPCK = pck[-1]
                self.save_model(best_pck=True)

    def get_bbox(self, meta):
        joints = meta['joints']
        joints = joints[joints[:, 0] > 0]
        if joints.size == 0:
            joints = np.array([[-1., -1., 0.]])

    def save_model(self, best_loss=False, best_pck=False):
        save_dir = self.save_root + time.strftime("%Y-%m-%d", time.localtime())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if best_loss:
            file_name = "/{0}_loss_{1}epoch.pt".format(round(self.best_loss, 3), self.epoch)
        elif best_pck:
            file_name = "/{0}_mPCK_{1}epoch.pt".format(round(self.best_mPCK, 3), self.epoch)
        else:
            raise NotImplementedError
        save_file = save_dir + file_name
        print(f"{save_file=}")

        self.save_dict["epoch"] = self.epoch
        self.save_dict["model_state"] = self.model.module.state_dict()
        self.save_dict["optimizer"] = self.optimizer.state_dict()
        torch.save(self.save_dict, save_file)

    def run(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            print('\nExp %s Epoch %d of %d @ %s' %
                  (exp_id, epoch + 1, self.end_epoch, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            self.epoch = epoch

            self.train()
            if epoch % cfg['eval_interval'] == 0:
                self.test()
        self.writer.close()


if __name__ == '__main__':
    f = Main()
    f.run()
