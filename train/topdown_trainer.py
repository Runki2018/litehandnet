import os
import torch
from collections import defaultdict
from utils.evaluation import evaluate_ap
from train.spawn_dist import all_gather_object, all_reduce



def save_file(output_path, save_dict, tag=None):
        if tag is None:
            file_name = "checkpoint.pth"
        else:
            file_name = 'best_model.pth' # f"{tag}.pth"
            
        save_file = output_path.joinpath(file_name)
        print(f"save checkpoint => {save_file}")
        torch.save(save_dict, save_file)

def _average_metric(sum_values, num_samples):
        if isinstance(sum_values, (list,)):
            return [v / num_samples * 100 for v in sum_values]
        return sum_values / num_samples * 100



@torch.no_grad()
def val_one_epoch(cfg, device, model, criterion, val_loader):
    # with_simdr = cfg.LOSS.with_simdr
    # num_samples = 0
    loss_dict = defaultdict(float)
    for meta in val_loader:
        img = meta['img']
        outputs = model(img.to(device, non_blocking=True))
        _, _loss_dict = criterion(outputs, meta)

        for k, v in _loss_dict.items():    
            loss_dict['sum'] += v
            loss_dict[k] += v

    return loss_dict


def warmup(step, warmup_steps, max_lr, device, model, criterion,
           train_loader, optimizer, fp16=False):
    """给模型预热, 防止模型训练前期波动太大, 导致难收敛
    https://blog.csdn.net/qq_42530301/article/details/124546903
    """
    for meta in train_loader: 
        img = meta['img']
        if step <= warmup_steps:
            lr = max_lr * step / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        step += 1

        outputs = model(img.to(device, non_blocking=True))
        loss, _ = criterion(outputs, meta)
        # compute gradient and do update step
        optimizer.zero_grad()
        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()
    return step

def train_one_epoch(cfg, device, model, criterion, train_loader, optimizer, fp16=False):
    loss_dict = defaultdict(float)
    for meta in train_loader: 
        img = meta['img']
        outputs = model(img.to(device, non_blocking=True))
        loss, _loss_dict = criterion(outputs, meta)

        # compute gradient and do update step
        optimizer.zero_grad()
        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()

        for k, v in _loss_dict.items():    
            loss_dict['sum'] += v
            loss_dict[k] += v

    return loss_dict


