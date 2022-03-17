import os
from numpy import dtype
import torch
from torch import gather, nn
import torch.multiprocessing as mp
import torch.distributed as dist
from utils.training_kits import set_seeds


def init_spawn_distributed(cfg, args):
    if args.FP16_ENABLED:
        assert torch.backends.cudnn.enabled, \
            "fp16 mode requires cudnn backend to be enabled."
        
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        else:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * args.ngpus_per_node + args.gpu
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.
              format(args.dist_url, args.world_size, args.rank))
        dist.init_process_group(
            backend='nccl',
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )


def model_cuda(cfg, args, model):
    if args.distributed:
        if cfg['syncBN']:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu],
                find_unused_parameters=cfg['find_unused_parameters']
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=cfg['find_unused_parameters'])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    return model
   
 
def all_reduce(values, device='cuda'):
    """
    https://pytorch.org/docs/master/distributed.html#torch.distributed.all_reduce
    """
    assert isinstance(values, (list, tuple)), "{} should be a list or tuple".format(values)  
    # define tensor on GPU, count and total is the result at each GPU
    _part_results = torch.tensor(values, dtype=torch.float32, device=device)
    
    # synchronizes all processes
    dist.barrier()
    # Reduces the tensor data across all machines in such a way that all get the final result.
    dist.all_reduce(_part_results, op=torch.distributed.ReduceOp.SUM)
    all_reults = _part_results.tolist()
    
    return all_reults

def all_gather_object(values, rank):
    """
    https://pytorch.org/docs/master/distributed.html#torch.distributed.all_gather
    """
    assert isinstance(values, (list, )), "{} should be a list".format(values)  
    # define the structure of final output
    output = [None for _ in range(dist.get_world_size())]
    gather_list = output.copy()
    gather_list[rank] = values
    # synchronizes all processes
    dist.barrier() 
    # gather the  data across all machines in such a way that all get the final result.
    dist.all_gather_object(output, gather_list)

    return output

    
    

    
    