from re import S
import torch
import argparse
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
from utils.CenterSimDRParser import ResultParser

from models.center_simDR import LiteHourglassNet as Network

from loss.loss import SimDRLoss as Loss
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
        self.device = torch.device(args.gpu)
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
        self.model = Network()
        
        self.criterion = Loss()

        pl = [p for p in self.model.parameters() if p.requires_grad] + \
             [p for p in self.criterion.parameters() if p.requires_grad]  # 学习MTL的loss的权重
        
        if cfg['optim'] == 'SGD':
            self.optimizer = optim.SGD(pl, lr=args.lr, momentum=0.9)
        elif cfg['optim'] == 'RMSprop':
            self.optimizer = optim.RMSprop(pl, lr=args.lr)
        else:
            self.optimizer = optim.AdamW(pl, lr=args.lr)
            # self.optimizer = optim.NAdam(pl, lr=args.lr)

        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min",   patience=20,
        #                                                       min_lr=cfg["lr"] * 0.001, cooldown=10)

        # 自定义的逐周期递减正弦学习率曲线
        T, lr_gamma, min_lr = cfg['T'], cfg['lr_gamma'], cfg['min_lr']
        lambda1 = lambda epoch: np.cos((epoch % (T + (epoch / T)) / (T + (epoch / T))) * np.pi / 2) * (lr_gamma ** (epoch / T)) + min_lr
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda1)

        # load checkpoint
        if cfg["reload"]:
            self.save_dict = torch.load(cfg["checkpoint"], map_location=torch.device('cpu'))
            if 'state_dict' in self.save_dict:
                state, is_match = load_pretrained_state(self.model.state_dict(),
                                                        self.save_dict['state_dict'])
            elif 'model_state' in self.save_dict:
                state, is_match = load_pretrained_state(self.model.state_dict(),
                                                        self.save_dict['model_state'])
            else:
                state, is_match = load_pretrained_state(self.model.state_dict(),
                                                        self.save_dict)

            print(f"reload checkpoint and is_match: {is_match}")
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
                os.makedirs(self.save_root, exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint_path)

            dist.barrier()
            # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
            self.model = self.model.to(self.device)
            # self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

        # 存储参数的字典
        self.best_loss = 0
        self.best_mPCK = 0
        self.best_ap = 0
        self.save_dict = {"epoch": 0, "lr": [], "loss": [], "mPCK": [], "ap": [],
                          "state_dict": {}, "optimizer": {}, "config": cfg}

        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                                            self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)
                                            
        if self.rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
            print(args)
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(logdir=self.save_root + 'log')  # 记录训练参数日志
            # self.n_out = len(cfg['param'])
            # assert self.n_out == cfg['nstack'] + cfg['higher_output']
            # print(f"{self.n_out=}")
        
        # 转为DDP模型, 这步放在最后，注意在DDP之前模型要先model.to(device)到相应设备
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                         device_ids=[args.gpu],
                                                         output_device=args.gpu)
                                                        #  ,find_unused_parameters=True)
        print("Model is ready!")
        
        for state in self.optimizer.state.values():
            for k, v in state.items():  # optimizer加载参数时,tensor默认在CPU上
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
                    
        self.result_parser = ResultParser()

    def train(self):
        self.model.train()
        # with stdout_to_tqdm() as save_stdout:
        if self.rank == 0:
            self.train_loader = tqdm(self.train_loader, desc="training ") 

        for meta in self.train_loader: 
            meta = [gt.to(self.device) for gt in meta]
            img, target_x, target_y, target_weight, centermap, centermask, bbox, gt_kpts = meta
            pred_centermap, pred_x, pred_y = self.model(img)
            loss, loss_lost = self.criterion(pred_centermap, centermap, centermask,
                                 pred_x, pred_y, target_x, target_y, target_weight)

            loss.backward()
            # loss = reduce_value(loss, average=True)
            self.optimizer.step()
            self.optimizer.zero_grad()

        with torch.no_grad():
            names = ['center_loss', 'wh_loss', 'offset_loss', 'kpts_loss']
            for name, loss_value in zip(names, loss_lost):
                print(f"{name}: {loss_value}")
            lr = self.optimizer.param_groups[0]['lr']
            if lr > 5e-7:
                    # self.scheduler.step(metrics=100 - self.best_mPCK)
                    self.scheduler.step()

            if self.rank == 0:
                loss = loss.item()
                print(exp_id, f"{loss=}")       
                print(f"{lr=}")

                # 记录训练数据
                self.save_dict["lr"].append(lr)
                self.save_dict["loss"].append(loss)

                best_loss = min(self.save_dict["loss"])

                self.writer.add_scalar("loss", loss, self.epoch)
                self.writer.add_scalar("lr", lr, self.epoch)
            
        # 等待所有进程计算完毕
        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)

    def test(self):
        self.model.eval()
        if self.rank == 0:
            with torch.no_grad():
                pck_list = []
                ap50_list = []
                ap_list = []
                for meta in tqdm(self.test_loader, desc="testing "):
                    meta = [gt.to(self.device) for gt in meta]
                    img, target_x, target_y, target_weight, centermap, centermask, bbox, gt_kpts = meta
                    pred_centermap, pred_x, pred_y = self.model(img)
                    
                    # 结果解析，得到原图关键点和边界框
                    pred_kpts, pred_bboxes = self.result_parser.parse(pred_centermap, pred_x, pred_y)
                    ap50, ap = self.result_parser.evaluate_ap(pred_bboxes, bbox)
                    ap50_list.append(ap50)
                    ap_list.append(ap)
                    
                    avg_pck = self.result_parser.evaluate_pck(pred_kpts, gt_kpts, thr=0.2)
                    pck_list.append(avg_pck * img.shape[0]) 
                    
                # 记录训练数据
                PCK = sum(pck_list) / self.test_set.__len__()
                ap_final = float(np.mean(ap_list))
                ap50_final = float(np.mean(ap50_list))
                print(f"{PCK=} \n AP = {ap_final}\n AP50 = {ap50_final}")

                # 记录训练数据
                self.save_dict["mPCK"].append(PCK)
                self.save_dict["ap"].append(ap_final)
                self.writer.add_scalar('mPCK', PCK, self.epoch)
                self.writer.add_scalars('AP', {"AP50": ap50_final, "AP": ap_final}, self.epoch)

                if len(self.save_dict["mPCK"]) > 3:
                    for i in range(-1, -4, -1):
                        print("pck{} = {:.5f}".format(i, self.save_dict["mPCK"][i]), end="\t")
                else:
                    print("pck = {:.5f}".format(PCK))

                if PCK > self.best_mPCK:
                    self.best_mPCK = PCK
                    self.save_model(best_pck=True)

                if ap_final > self.best_ap:
                    self.best_ap = ap_final
                    self.save_model(best_ap=True)

        # 等待所有进程计算完毕
        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)

    def save_model(self, best_loss=False, best_pck=False, best_ap=False):
        save_dir = self.save_root + datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if best_loss:
            file_name = "/{0}_loss_{1}epoch.pt".format(round(self.best_loss, 3), self.epoch)
        elif best_pck:
            file_name = "/{0}_PCK_{1}epoch.pt".format(round(self.best_mPCK, 3), self.epoch)
        elif best_ap:
            file_name = "/{0}_AP_{1}epoch.pt".format(round(self.best_ap, 3), self.epoch)
        else:
            raise NotImplementedError

        save_file = save_dir + file_name
        print(f"{save_file=}")

        self.save_dict["epoch"] = self.epoch
        self.save_dict["state_dict"] = self.model.state_dict()
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
    # CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 29501 train_distributed.py

    # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_distributed_center_simdr.py
