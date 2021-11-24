from re import S
import torch
import argparse
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datetime import datetime
from utils.transforms import get_final_preds
from models.ResultAttention import RANet as Network
from loss.loss import HMLoss as Loss
from train.distributed_utils import init_distributed_mode, dist, cleanup, reduce_value, reduce_value
from utils.training_kits import stdout_to_tqdm, load_pretrained_state
from config.config import config_dict as cfg
import os
from data import get_dataset


os.environ['CUDA_VISIBLE_DEVICES'] = cfg["CUDA_VISIBLE_DEVICES"]
exp_id = cfg["experiment_id"]

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



def get_argment():
    parser = argparse.ArgumentParser(description="Define DDP training parameters.")
    parser.add_argument('--epochs', type=int, default=cfg['n_epochs'])
    parser.add_argument('--batch-size', type=int, default=cfg['batch_size'])
    parser.add_argument('--lr', type=float, default=cfg['lr'])  # 初始学习率cosine
    # 是否启用SyncBatchNorm, 启用后会降低训练速度，建议前期为False，在训练后期再开启。
    parser.add_argument('--syncBN', type=bool, default=cfg['syncBN'])

    parser.add_argument('--weights', type=str, default=cfg["checkpoint"],
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


class Main:
    """
        # 一机多卡训练
    """

    def __init__(self, args):
        # 定义参数
        self.epoch = 0  # 当前训练的epoch
        self.start_epoch = 0  # 开始的epoch，如果是重新开始训练为0，如果是继续训练则接着上次训练的epoch开始
        self.end_epoch = args.epochs # 训练结束的epoch
        self.save_root = cfg["save_root"] + exp_id + "/"


        # 初始化各进程环境
        init_distributed_mode(args=args)

        self.rank = args.rank
        self.device = torch.device(args.device)
        self.batch_size = args.batch_size
        self.weights_path = args.weights
        args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增

        # 数据集和数据加载器
        # 测试集只在一个GPU上使用
        self.test_set, self.test_loader = get_dataset('test', just_dataset=False)
        # 将训练集按GPU数划分
        train_set = get_dataset('train', just_dataset=True)  
        # 给每个 rank 对于的进程分配训练样本索引
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

        # 将样本索引每batch_size个元素组成一个list
        train_batch_sampler = torch.utils.data.BatchSampler(
                            self.train_sampler, self.batch_size, drop_last=True)
                            
        num_wokers = min([os.cpu_count(),
                        self.batch_size if self.batch_size > 1 else 0,
                        cfg['workers']])

        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_sampler=train_batch_sampler,
                                                pin_memory=True,
                                                num_workers=num_wokers)

        # 多GPU训练
        self.model = Network().to(self.device)
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=20,
                                                              min_lr=cfg["lr"] * 0.001, cooldown=10)

        self.criterion = Loss()
        for state in self.optimizer.state.values():
            for k, v in state.items():  # optimizer加载参数时,tensor默认在CPU上
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

        # load checkpoint
        if cfg["reload"]:
            self.save_dict = torch.load(cfg["checkpoint"])
            if 'model_state' in self.save_dict:
                state, is_match = load_pretrained_state(self.model.state_dict(),
                                                        self.save_dict['model_state'])
            elif 'state_dict' in self.save_dict:
                state, is_match = load_pretrained_state(self.model.state_dict(),
                                                        self.save_dict['state_dict'])
            else:
                state, is_match = load_pretrained_state(self.model.state_dict(),
                                                        self.save_dict)
            self.model.load_state_dict(state)
            if not cfg["just_model"] and is_match:
                self.optimizer.load_state_dict(self.save_dict["optimizer"])
                self.start_epoch = self.save_dict["epoch"]
                self.best_loss = min(self.save_dict["loss"])
                self.best_mPCK = max(self.save_dict["mPCK"])
        else:
            checkpoint_path = self.save_root + "initial_weights.pt"
            # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
            if self.rank == 0 and not os.path.exists(checkpoint_path):
                torch.save(self.model.state_dict(), checkpoint_path)

            dist.barrier()
            # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        # 存储参数的字典
        self.best_loss = 0
        self.best_mPCK = 0
        self.best_ap = 0
        self.save_dict = {"epoch": 0, "lr": [], "loss": [], "mPCK": [], "ap": [],
                          "model_state": {}, "optimizer": {}, "config": cfg}

        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                                            self.model).to(self.device)

        if self.rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
            print(args)
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(logdir=self.save_root + 'log')  # 记录训练参数日志
        
        # 转为DDP模型, 这步放在最后
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                         device_ids=[args.gpu],
                                                         find_unused_parameters=True)
        print("Model is ready!")

    def train(self):
        self.model.train()
        with stdout_to_tqdm() as save_stdout:
            if self.rank == 0:
                self.train_loader = tqdm(self.train_loader, desc="training ") 

            for img, target, target_weight, _ in self.train_loader:
                
                # torch.cuda.synchronize()
                # s = time.time()
                output = self.model(img.to(self.device))
                # torch.cuda.synchronize()
                # e = time.time()
                # print(f'time = {e-s}')

                target = target.to(self.device)
                target_weight = target_weight.to(self.device)
                loss = self.criterion(output, target, target_weight)

                loss.backward()
                # loss = reduce_value(loss, average=True)
                self.optimizer.step()
                self.optimizer.zero_grad()

        with torch.no_grad():
            lr = self.optimizer.param_groups[0]['lr']
            if lr > 5e-7:
                    self.scheduler.step(metrics=loss)

            if self.rank == 0:
                loss = loss.item()
                print(exp_id, f"{loss=}")       
                print(f"{lr=}")

                # 记录训练数据
                self.save_dict["lr"].append(lr)
                self.save_dict["loss"].append(loss)

                best_loss = min(self.save_dict["loss"])
                if loss <= best_loss:
                    self.best_loss = loss
                    self.save_model(best_loss=True)

                self.writer.add_scalar("loss", loss, self.epoch)
                self.writer.add_scalar("lr", lr, self.epoch)
            
            # 等待所有进程计算完毕
            if self.device != torch.device("cpu"):
                torch.cuda.synchronize(self.device)

    def test(self):
        self.model.eval()
        if self.rank == 0:
            with torch.no_grad():
                n_out = cfg['nstack'] + 1
                pck = [0] * n_out
                pred_kpts = []
                for img, target, _, meta in tqdm(self.test_loader, desc="testing "):
                    hm_list = self.model(img.cuda())
                    c = meta['center'].numpy()
                    s = meta['scale'].numpy()
                    for i in range(n_out):
                        if len(pred_kpts) == i:
                            pred_kpts.append([])
                        kpt = get_final_preds(hm_list[i], c, s)
                        # kpt = get_final_preds(target, c, s)
                        pred_kpts[i].append(kpt)

                for i in range(n_out):
                    preds = np.concatenate(pred_kpts[i])  # [n_images, n_joints, 3]
                    print(f'{preds.shape=}')
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

        # 等待所有进程计算完毕
        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)

    def save_model(self, best_loss=False, best_pck=False):
        save_dir = self.save_root + datetime.now().strftime("%Y-%m-%d")
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
        self.save_dict["model_state"] = self.model.state_dict()
        self.save_dict["optimizer"] = self.optimizer.state_dict()
        torch.save(self.save_dict, save_file)

    def run(self):
        try:
            for epoch in range(self.start_epoch, self.end_epoch):
                if self.rank == 0:
                    print('\nExp %s Epoch %d of %d @ %s' %
                        (exp_id, epoch + 1, self.end_epoch,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                self.epoch = epoch

                self.train_sampler.set_epoch(epoch)
                self.train()
                if (epoch % cfg['eval_interval']) == 0:
                    self.test()
            if self.rank == 0:
                self.writer.close()
        finally:
            cleanup()  # 销毁所有进程，释放资源


if __name__ == '__main__':
    # 创建多个进程，每个进程解析一次参数和创建一个main类实例
    opt = get_argment()
    Main(opt).run()
    # python -m torch.distributed.launch --nproc_per_node=4 --use_env train_distributed.py
