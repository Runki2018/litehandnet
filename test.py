from pathlib import Path
import json
import torch
import argparse
from tqdm import tqdm
from config import get_config
from models import get_model
from utils.misc import get_checkpoint_path, get_output_path, get_pickle_path
from datasets.dataloader import make_dataloader
from utils.training_kits import load_pretrained_state
from utils.post_processing.decoder import TopDownDecoder
from utils.post_processing.vis_results import SaveResultImages


def get_argument():
    parser = argparse.ArgumentParser(description="Define DDP training parameters.")
    parser.add_argument('--cfg', default='config/freihand/cfg_freihand_hg_ms_att.py', 
                        help='experiment configure file path')
    
    parser.add_argument('--load_best', action='store_true',
                        help='load the best model pth ')
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
    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='test the train set')

    opt = parser.parse_args()
    return opt

def print_metric(name_dict):
    for k, v in name_dict.items():
        print(f'=> {k:5}:{v:.4f}')

def save_metric(name_dict, output_path, is_best=True):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    if is_best:
        output_file = output_path.joinpath('best_pth_metric.json')
    else:
        output_file = output_path.joinpath('checkpoint_pth_metric.json')
    with output_file.open('w') as fd:
        json.dump(name_dict, fd, indent=True)

def test():
    """
        # ??????????????????
    """
    args = get_argument()
    cfg = get_config(args.cfg)

    # ???????????????????????????
    if args.train:
        cfg.DATASET.test.ann_file = cfg.DATASET.train.ann_file
        cfg.DATASET.test.img_prefix = cfg.DATASET.train.img_prefix
    test_set, test_loader = make_dataloader(cfg, data_type='test', use_cpu=True)
    # ???GPU??????
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
            
    # ????????????
    output_path = get_output_path(cfg, args.cfg)
    # load checkpoint
    cfg.CHECKPOINT['load_best'] = args.load_best
    print(f"{args.load_best=}")
    print(f"{cfg.DATASET.test.ann_file=}")
    checkpoint_file = get_checkpoint_path(cfg, output_path)
    if checkpoint_file.exists():
        # reload the last checkpoint
        save_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        epoch = save_dict['epoch']
        state_dict = save_dict['state_dict']
        state_dict, is_match = load_pretrained_state(model.state_dict(), state_dict)
        print(f"reload model => {checkpoint_file}\n=> {epoch=}\t{is_match=}")
        assert is_match, "model pth not match the code of current model!"
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"model not exist! {checkpoint_file}")

    vis_tools = SaveResultImages(test_set, output_path)
    result_decoder = TopDownDecoder(cfg)

    if 'litehandnet' in cfg.MODEL.name.lower():
        model.module.deploy_model()
# --------------------------- test -------------------------------------
    # train_sampler.set_epoch(epoch)
    test_loader = tqdm(test_loader, desc="testing ...") 
    model.eval()
    results = []
    simdr_results = []
    for meta in test_loader:
        img = meta['img']
        outputs = model(img.to(device))
        if cfg.MODEL.name.lower() == 'srhandnet':
            outputs = outputs[-1]
        # elif 'hourglass' in cfg.MODEL.name.lower():
        #     outputs = outputs[:, -1]

        # outputs = meta['target']  # ??????GT?????????????????????100%??????
        results.append(result_decoder.decode(meta, outputs))

        if result_decoder.k > 0:  # with simdr == True
            simdr_results.append(result_decoder.decode_simdr(meta, outputs))

    name_value = test_set.evaluate(results, output_path, cfg.EVAL.metric)
    print(f"{len(results)=}")
    print(f"Heatmap:")
    print_metric(name_value)
    save_metric(name_value, output_path, args.load_best)

    if result_decoder.k > 0:  # with simdr == True
        name_value = test_set.evaluate(simdr_results, output_path, ['AUC'])
        print(f"{len(results)=}")
        print(f"SimDR:")
        print_metric(name_value)
        vis_tools.save_images_with_joints(meta['img'], simdr_results[-1]['preds'],
                                    meta['joints_3d_visible'], 'gt_simdr_joints.jpg')

    if cfg.MODEL.name.lower() == 'srhandnet':
        target = meta['target'][-1]
    elif 'hourglass' == cfg.MODEL.name.lower():
        target = meta['target'][:, -1]
    else:
        target = meta['target']

    vis_tools.save_images_with_joints(meta['img'], meta['joints_3d'],
                                      meta['joints_3d_visible'], 'gt_joints.jpg')
    vis_tools.save_images_with_joints(meta['img'], results[-1]['hm_preds'],
                                      meta['joints_3d_visible'], 'pred_joints.jpg')
    vis_tools.save_images_with_heatmap(meta['img'], target, 'gt_heatmap.jpg')
    vis_tools.save_images_with_heatmap(meta['img'], outputs, 'pred_heatmap.jpg')

if __name__ == '__main__':
    test()