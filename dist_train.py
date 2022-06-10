# from data import get_dataset
# import data
# from loss.loss import MultiTaskLoss as Loss
# from utils.result_parser import ResultParser
# ---------------------------------------------

import os
import torch
import argparse
from copy import deepcopy
from shutil import copy2
from tqdm import tqdm
from datetime import datetime
from loss import get_loss
from config import get_config
from models import get_model
from datasets.dataloader import make_dataloader
from train.optimizer_scheduler import get_optimizer, get_scheduler
from utils.post_processing.decoder import TopDownDecoder
from utils.training_kits import load_pretrained_state
from train.topdown_trainer import train_one_epoch, val_one_epoch, save_file, warmup
from utils.misc import get_checkpoint_path, get_output_path, get_pickle_path

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

def main(gpu, cfg, args):
    """
        # 一机多卡训练
    """
    args.gpu = gpu  # 系统自动分配的参数：gpu号
    device = torch.device(args.gpu)
   
    init_spawn_distributed(args)

    # 数据集和数据加载器
    val_set, val_loader = make_dataloader(cfg, data_type='val')
    train_set, train_loader = make_dataloader(cfg, data_type='train')

    # 多GPU训练
    model = get_model(cfg)   
    model = model_cuda(cfg, args, model)
    criterion = get_loss(cfg)
    
    cfg.OPTIMIZER.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    optimizer = get_optimizer(cfg, model, criterion) 
     
    if args.FP16_ENABLED:
        model = network_to_half(model)
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=1.0,
            dynamic_loss_scale=True
        )

    # 定义参数
    exp_name = cfg.DATASET.name + '_' + cfg.MODEL.name  # experiment name

    begin_epoch = 0  # 开始的epoch
    min_val_loss = 1e6
    end_epoch = cfg.TRAIN.total_epoches  # 训练结束的epoch
    output_path = get_output_path(cfg, args.cfg)
    

    # load checkpoint
    checkpoint_file = get_checkpoint_path(cfg, output_path)
    if cfg.CHECKPOINT.resume and checkpoint_file.exists():
        # reload the last checkpoint
        save_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        state_dict = save_dict['state_dict']
        state_dict, is_match = load_pretrained_state(model.state_dict(), state_dict)
        if args.rank == 0:
            print(f"reload checkpoint and {is_match=}")
        model.load_state_dict(state_dict)

        min_val_loss = save_dict.get('min_val_loss', 1e6)
        if is_match:
            begin_epoch = save_dict.get('epoch', 0)
            if save_dict['config'].OPTIMIZER.type == cfg.OPTIMIZER.type:
                optimizer.load_state_dict(save_dict["optimizer"])
                for state_dict in optimizer.state.values():
                    # optimizer加载参数时,tensor默认在CPU上
                    for k, v in state_dict.items():  
                        if torch.is_tensor(v):
                            state_dict[k] = v.to(device)
        save_dict['config'] = cfg
    else:
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        temp_file = output_path.joinpath('temp.pth')
        if args.rank == 0 and not temp_file.exists():
            os.makedirs(output_path, exist_ok=True)
            torch.save(model.state_dict(), temp_file)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        load_dict = torch.load(temp_file, map_location=torch.device('cpu'))
        state_dict = load_dict.get('state_dict', load_dict)
        model.load_state_dict(state_dict)

    
    scheduler = get_scheduler(optimizer, args.FP16_ENABLED, begin_epoch)

    if args.rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        from tensorboardX import SummaryWriter
        log_dir = output_path.joinpath('log')
        copy2(args.cfg, output_path.joinpath('config.py'))  # 保存配置文件
        if not log_dir.exists(): 
            log_dir.mkdir(mode=0o777)
        writer = SummaryWriter(logdir=str(log_dir))  # 记录训练参数日志
        if not cfg.CHECKPOINT.resume:
            temp_file = output_path.joinpath('temp.pth')
            temp_file.unlink(missing_ok=True)  # 删除初始化的权重文件
    else:
        writer = None

    
    max_lr = cfg.OPTIMIZER.lr
    warmup_steps = min(cfg.OPTIMIZER.warmup_steps, train_set.__len__())
    step = 1

    model.train()
    while step < warmup_steps:
        if args.rank == 0:
            print(f"warmup:{step}/{warmup_steps}")
            train_loader = tqdm(train_loader, desc="warm up ...") 
        step = warmup(step, warmup_steps, max_lr, device, model, criterion,
                      train_loader, optimizer, args.FP16_ENABLED)

    result_decoder = TopDownDecoder(cfg)
    for epoch in range(begin_epoch, end_epoch):  
        if args.rank == 0:
            print('\nExp %s Epoch %d of %d @ %s' %
                (exp_name, epoch + 1, end_epoch,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# ---------------------------- train -------------------------------------
        # ! 是否需要加上这个，HigherHRNet没用这个
        # train_sampler.set_epoch(epoch)
        if args.rank == 0:
            train_loader = tqdm(train_loader, desc="training ...") 

        model.train()
        train_loss_dict = train_one_epoch(cfg, device, model, criterion,
                                          train_loader, optimizer, args.FP16_ENABLED)    
        scheduler.step()
 
        # 等待所有进程计算完毕
        # if device != torch.device("cpu"):
        #     torch.cuda.synchronize(device)

# ---------------------------- test -------------------------------------
        if (epoch % cfg.EVAL.interval == 0):
            model.eval()
            if args.rank == 0:  # 在主节点显示进度条
                val_loader = tqdm(val_loader, desc="testing ...")
 
            val_loss_dict = val_one_epoch(cfg, device, model, criterion, val_loader)

            # 等待所有进程计算完毕
            # if device != torch.device("cpu"):
            #     torch.cuda.synchronize(device)

            if args.rank == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"{lr=}")
                print("train loss".center(30, '-'))
                for name, loss_value in train_loss_dict.items():
                    print(f"{name:8}: {loss_value:.3f}", end='\t')
                print()
                print("val loss".center(30, '-'))
                for name, loss_value in val_loss_dict.items():
                    print(f"{name:8}: {loss_value:.3f}", end='\t')
                print()

                # 记录训练数据
                writer.add_scalars("train_loss", train_loss_dict, epoch)
                writer.add_scalars("val_loss", val_loss_dict, epoch)
                writer.add_scalar("lr", lr, epoch)   

                if min_val_loss > val_loss_dict['sum']: 
                    min_val_loss = val_loss_dict['sum']
                    # 记录训练数据
                    save_dict = {
                        "epoch": epoch,
                        'min_val_sum': min_val_loss,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": cfg
                        }

                    save_file(output_path, save_dict, tag='min_val_loss')

        # save checkpoint.pth
        if args.rank == 0:
            if (epoch % cfg.CHECKPOINT.interval == 0) \
                or (epoch == end_epoch - 1):
                
                save_dict = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                }
                save_file(output_path, save_dict)

    if args.rank == 0:
        writer.close()


def boot():
    # 单机多卡
    args = get_argument()
    cfg = get_config(args.cfg)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.TRAIN.CUDA_VISIBLE_DEVICES)
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

    args.distributed = args.world_size > 1 or cfg.TRAIN.distributed
    args.ngpus_per_node = torch.cuda.device_count()
    cfg['TRAIN']['num_gpus'] = args.ngpus_per_node

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
                args=(cfg, args),
                join=True
            )
        except KeyboardInterrupt:
            # dist.destroy_process_group()  # 训练完成后记得销毁进程组释放资源
            torch.cuda.empty_cache()
        finally:
            torch.cuda.empty_cache()

    else:
        # Simply call main_worker function
        main(
            str(cfg.TRAIN.CUDA_VISIBLE_DEVICES),
            cfg,
            args,
        )

if __name__ == '__main__':
    boot()
    # python dist_train.py --cfg config/freihand/cfg_freihand_hg_ms_att.py
