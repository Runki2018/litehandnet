from datetime import datetime
import torch
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.evaluation import evaluate_ap
from train.spawn_dist import all_gather_object, all_reduce

def save_file(save_root, save_dict, value=0,
              is_best_pck=False, is_best_ap=False):
        save_dir = save_root + datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if is_best_pck:
            file_name = "best_pck_{}.pt".format(round(value, 4))     
        elif is_best_ap:
            file_name = "best_ap_{}.pt".format(round(value, 4))
        else:
            file_name = "checkpoint.pt"

        save_file = save_dir + file_name
        print(f"{save_file=}")
        torch.save(save_dict, save_file)

def _average_metric(sum_values, num_samples):
        if isinstance(sum_values, (list,)):
            return [v / num_samples * 100 for v in sum_values]
        return sum_values / num_samples * 100

def test_one_epoch(cfg, rank, epoch, model, dataloader, num_samples,
                   result_parser, device, writer, fp16=False):
    
    NUM_OUT = len(cfg['hm_loss_factor'])
    metrics = dict( 
                ap=[0] * NUM_OUT,
                ap50=[0] * NUM_OUT,
                coor_pck=[0] * NUM_OUT,
                )
    
    pck_thr = cfg['pck_thr']
    with_region_map = cfg['with_region_map']
    with_cycle_detection = cfg['with_cycle_detection']
    with_simdr = cfg['simdr_split_ratio'] > 0
    
    for meta in dataloader:
        if with_simdr:
            img, targets, target_weight, bbox, gt_kpts, target_x, target_y = meta
            outputs, pred_x, pred_y = model(img.to(device))
        else:
            img, targets, target_weight, bbox, gt_kpts = meta
            outputs = model(img.to(device))
        
        if cfg['model'] == 'srhandnet':
                outputs = [outputs[-1]]  # 只计算最后一个输出的性能指标
                metrics = dict(ap=[0],ap50=[0],coor_pck=[0])
                
        num_img = img.shape[0]
        for i, output_pred in enumerate(outputs):
            if with_region_map:
                ap50, ap, _ = \
                    evaluate_ap(output_pred[:, -3:], bbox, cfg['image_size'][0])
                metrics['ap50'][i] += ap50 * num_img
                metrics['ap'][i] += ap * num_img
                
                if with_cycle_detection:
                    # 结果解析，得到原图关键点和边界框
                    pred_bboxes = result_parser.get_pred_bbox(output_pred[:, -3:]) 
                    pred_kpts = result_parser.get_group_keypoints(
                                            model, img, pred_bboxes,output_pred[:, :-3])
                else:
                    pred_kpts = result_parser.get_pred_kpt(
                        output_pred[:, :-3], resized=True)   
                    pred_kpts = pred_kpts[:, None]
            else:
                pred_kpts = result_parser.get_pred_kpt(output_pred, resized=True)
                pred_kpts = pred_kpts[:, None]
                
            metrics['coor_pck'][i] += \
                result_parser.evaluate_pck(pred_kpts, gt_kpts, bbox, pck_thr) * num_img  
                 
    for i, (metric_name, sum_values) in enumerate(metrics.items()):
        # metrics[metric_name] = all_gather_object(sum_values, rank)
        metrics[metric_name] = all_reduce(sum_values, device)     
        metrics[metric_name] = _average_metric(metrics[metric_name] , num_samples)
        
    if rank == 0:
        for i, (k, v) in enumerate(metrics.items()):
                    print("_ {} _\t{} = {}".format(i, k, v))
                    
        writer.add_scalars(
            'mPCK', { f"pck_{i}": v for i, v in enumerate(metrics['coor_pck'])}, epoch)
        
        writer.add_scalars(
            'AP', {f"AP_{i}": v for i, v in enumerate(metrics['ap'])}, epoch)

        writer.add_scalars(
            'AP50', {f"AP50_{i}": v for i, v in enumerate(metrics['ap50'])}, epoch)
    
    return metrics['coor_pck'][-1], metrics['ap'][-1], metrics['ap50'][-1]


def train_one_epoch(cfg, model, criterion, train_loader, train_set,
                    optimizer, device, fp16=False):
    with_simdr = cfg['simdr_split_ratio'] > 0
    
    loss_sum = 0
    loss_dict = defaultdict(float)
    for meta in train_loader: 
        
        for i, gt in enumerate(meta):
            if isinstance(gt, (list,)):
                meta[i] = [m.to(device) for m in meta[i]]
            else:
                meta[i] = gt.to(device)
        
        if with_simdr:
            img, targets, target_weight, bbox, gt_kpts, target_x, target_y = meta
            outputs, pred_x, pred_y = model(img)
            loss, _loss_dict = criterion(outputs, targets, target_weight, False,
                                        pred_x, pred_y, target_x, target_y)
        else:
            img, targets, target_weight, bbox, gt_kpts = meta
            outputs = model(img)
            loss, _loss_dict = criterion(outputs, targets, target_weight, False)

        # compute gradient and do update step
        optimizer.zero_grad()
        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        
        for k, v in _loss_dict.items():    
            loss_sum += v
            loss_dict[k] += v
            
    return loss_sum, loss_dict


        
    
    
    
 