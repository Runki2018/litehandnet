from collections import defaultdict
import os
from sched import scheduler
import torch
import argparse
from tqdm import tqdm
from datetime import datetime

from data import get_dataset
from config import get_config
from models.pose_estimation import get_model
from train.optimizer_scheduler import get_optimizer, get_scheduler
from train.freihand_trainer import train_one_epoch, test_one_epoch, save_file
from loss.loss import MultiTaskLoss as Loss

from utils.result_parser import ResultParser
from utils.training_kits import load_pretrained_state

import torch.distributed as dist
import torch.multiprocessing as mp
from train.spawn_dist import init_spawn_distributed, model_cuda
from train.fp16_utils.fp16util import network_to_half
from train.fp16_utils.fp16_optimizer import FP16_Optimizer


def get_argument():
    parser = argparse.ArgumentParser(description="Define DDP training parameters.")
    parser.add_argument('--cfg', default='config/freihand/cfg_freihand_hg_ms_att.py', help='experiment configure file path')
    parser.add_argument('--FP16-ENABLED', type=bool, default=False,
                        help='Mixed-precision training')
    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:65432',
                        # default="env://",
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    opt = parser.parse_args()
    return opt

def main(gpu, cfg, DATASET, args):
    """
        # 一机多卡训练
    """
    args.gpu = gpu
    init_spawn_distributed(cfg, args)
 
    exp_name = cfg["dataset"] + '__' + cfg['model']  # experiment name
        
    # 定义参数
    epoch = 0  # 当前训练的epoch
    begin_epoch = -1  # 开始的epoch，如果是重新开始训练为0，如果是继续训练则接着上次训练的epoch开始
    end_epoch = cfg['end_epoch'] # 训练结束的epoch
    save_root = cfg["save_root"] + exp_name + "/"

    device = torch.device(args.gpu)
    args.lr = cfg['lr'] * args.world_size  # 学习率要根据并行GPU的数量进行倍增

    # 数据集和数据加载器
    test_set, test_loader = get_dataset(cfg, DATASET, is_train=False, distributed=True)
    train_set, train_loader = get_dataset(cfg, DATASET, is_train=True, distributed=True)
    
    # 多GPU训练
    model = get_model(cfg)   
    model = model_cuda(cfg, args, model)
    
    criterion = Loss(cfg)
    
    optimizer = get_optimizer(cfg, model, criterion)  
    if args.FP16_ENABLED:
        model = network_to_half(model)
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=1.0,
            dynamic_loss_scale=True
        )

    # 存储参数的字典
    best_loss = 0
    best_PCK = 0
    best_AP = 0
    save_dict = {}
    
    # load checkpoint
    checkpoint_file = os.path.join(save_root, "checkpoint.pt")
    if cfg["reload"]:
        # reload the specified checkpoint
        if os.path.exists(cfg['checkpoint']):
            save_dict = torch.load(cfg["checkpoint"], map_location=torch.device('cpu'))
        # reload the last checkpoint    
        elif os.path.exists(checkpoint_file):
            save_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        
        state_dict = save_dict['state_dict'] if 'state_dict' in save_dict else save_dict
        state_dict, is_match = load_pretrained_state(model.state_dict(), state_dict)
        if args.rank == 0:
            print(f"reload checkpoint and is_match: {is_match}")
        
        model.load_state_dict(state_dict)
        if is_match:
            optimizer.load_state_dict(save_dict["optimizer"])
            begin_epoch = save_dict["epoch"]
            
    else:
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if args.rank == 0 and not os.path.exists(checkpoint_file):
            os.makedirs(save_root, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_file)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(
            torch.load(checkpoint_file, map_location=torch.device('cpu'))
            )
    for state_dict in optimizer.state.values():
        for k, v in state_dict.items():  # optimizer加载参数时,tensor默认在CPU上
            if torch.is_tensor(v):
                state_dict[k] = v.to(device)
    
    scheduler = get_scheduler(cfg, optimizer, args.FP16_ENABLED, begin_epoch)
     
    if args.rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(logdir=save_root + 'log')  # 记录训练参数日志
        if not cfg['reload']:
            os.remove(checkpoint_file)
    else:
        writer = None
                
    result_parser = ResultParser(cfg)
    for epoch in range(begin_epoch, end_epoch):  
        if args.rank == 0:
            loss_sum, loss_dict, pck, ap, ap50 = \
                None, None, None , None, None
            print('\nExp %s Epoch %d of %d @ %s' %
                (exp_name, epoch + 1, end_epoch,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# ---------------------------- train -------------------------------------
        # ! 是否需要加上这个，HigherHRNet没用这个
        # train_sampler.set_epoch(epoch)
        if args.rank == 0:
            train_loader = tqdm(train_loader, desc="training ...") 

        model.train()
        loss_sum, loss_dict = train_one_epoch(
            cfg, model, criterion, train_loader,train_set,
            optimizer, device, args.FP16_ENABLED)    
        
        scheduler.step()
        
        # 等待所有进程计算完毕
        # if device != torch.device("cpu"):
        #     torch.cuda.synchronize(device)

# ---------------------------- test -------------------------------------
        if (epoch % cfg['eval_interval']) == 0:
            model.eval()
            if args.rank == 0:
                test_loader = tqdm(test_loader, desc="testing ...")    
            with torch.no_grad():                 
                pck, ap, ap50 = test_one_epoch(
                    cfg, args.rank, epoch, model, test_loader,
                    test_set.__len__(), result_parser, device,
                    writer, args.FP16_ENABLED
                )

            # 等待所有进程计算完毕
            # if device != torch.device("cpu"):
            #     torch.cuda.synchronize(device)
                
            if args.rank == 0:
                if loss_sum is not None:
                    for name, loss_value in loss_dict.items():
                        print(f"{name}: {loss_value}")
                        
                    print(exp_name, "loss_sum = {}".format(loss_sum))       
                    lr = optimizer.param_groups[0]['lr']
                    print(f"{lr=}")

                    # 记录训练数据
                    writer.add_scalar("loss", loss_sum, epoch)
                    writer.add_scalar("lr", lr, epoch)   
                    
                if pck is not None: 
                                        # 记录训练数据
                    print("coor_pck = {:.3f}".format(pck))
                    print("ap = {:.3f}".format(ap))
                    print("ap50 = {:.3f}".format(ap50))

                    save_dict = {
                        "epoch": epoch,
                        "PCK": pck,
                        "AP": ap,
                        "AP50": ap50,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": cfg
                        }
                    
                    if pck > best_PCK:
                        best_PCK = pck
                        save_file(save_root, save_dict, pck, is_best_pck=True)

                    if ap > best_AP:
                        best_AP = ap
                        save_file(save_root, save_dict, ap, is_best_ap=True)
        
        # save last epoch file
        if args.rank == 0 and epoch == end_epoch - 1:
                save_dict = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg
                }
                save_file(save_root, save_dict)

    if args.rank == 0:
        writer.close()



def boot():
    # 单机多卡
    args = get_argument()
    cfg, DATASET = get_config(args.cfg)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['CUDA_VISIBLE_DEVICES']
    # args.ngpus_per_node = torch.cuda.device_count()
    # args.world_size = args.ngpus_per_node
    # if args.dist_url == "env://":
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '23456'
    
    if args.gpu is not None:
        print('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or cfg['distributed']
    args.ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size *= args.ngpus_per_node
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        try:
            mp.spawn(
                main,
                nprocs=args.ngpus_per_node,
                args=(cfg, DATASET, args),
                join=True
            )
        except KeyboardInterrupt:
            # dist.destroy_process_group()  # 训练完成后记得销毁进程组释放资源
            torch.cuda.empty_cache()

    else:
        # Simply call main_worker function
        main(
            cfg['CUDA_VISIBLE_DEVICES'],
            cfg,
            DATASET,
            args,
        )
    
if __name__ == '__main__':
    boot()
    # python train_distributed_freihand.py --cfg config/freihand/cfg_freihand_hg_ms_att.py
