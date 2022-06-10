import os
from pathlib import Path
import addict

def _mkdir(path: Path):
    if not isinstance(path, Path): path = Path(path)
    if not path.exists():  path.mkdir(mode=0o777, parents=True, exist_ok=True)

def get_output_path(cfg, cfg_path):
    if not isinstance(cfg, addict.Dict):
        cfg = addict.Dict(cfg)
    
    cfg_file_id = str(Path(cfg_path).stem).split('_')[1]
    # 检查配置文件命名id和配置中的ID是否一致
    assert cfg.ID == int(cfg_file_id), f"Error: {cfg.ID=}, but {cfg_file_id=}"
    
    # "root/dataset/model/cfg_stem/"
    save_root = Path(cfg.CHECKPOINT.save_root)
    save_dataset_dir = save_root.joinpath(cfg.DATASET.name)
    save_model_dir = save_dataset_dir.joinpath(cfg.MODEL.name)
    save_id_dir = save_model_dir.joinpath(cfg_file_id)
    
    _mkdir(save_root)
    _mkdir(save_dataset_dir)
    _mkdir(save_model_dir)
    _mkdir(save_id_dir)

    return save_id_dir

def get_checkpoint_path(cfg, output_path):
    if cfg.CHECKPOINT.load_best:
        return output_path.joinpath('best_model.pth')
    return output_path.joinpath('checkpoint.pth')

def get_pickle_path(output_path, rank):
    return output_path.joinpath(f'results_rank{str(rank)}.pkl')



