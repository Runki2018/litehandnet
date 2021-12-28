import contextlib
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm


def set_seeds(seed=1, cuda_deterministic=True):
    """
        作用：固定住深度模型训练的过程，使得每次从头开始训练模型初始化方式和数据读取方式保持一致，
        便于分析网络模型的性能。
    @param seed: int 随机数种子
    @return: None
    """
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True  # 保证每次卷积算法返回结果一样
        torch.backends.cudnn.benchmark = False  # 保证每次cudnn使用的都是同一种算法，而不是自行选择最优算法
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def load_pretrained_state(model_state, pretrained_state):
    """根据保持模型参数的关键字进行参数加载， 两个模型的不需要完全一致"""
    keys1 = model_state.keys()
    keys2 = pretrained_state.keys()
    # print(f"{keys2=}")
    fully_match = True  # 两个模型参数字典是否完全一致
    i = 0
    for k1 in keys1:
        for k2 in keys2:
            if k1 in k2 and model_state[k1].shape == pretrained_state[k2].shape:
                # print(f"{k1} = {k2}")
                model_state[k1] = pretrained_state[k2]
                i += 1
    if i != len(keys1):
        fully_match = False
    return model_state, fully_match


class TqdmFile(object):
    dummy_file = None

    def __init__(self, dummy_file):
        self.dummy_file = dummy_file

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.dummy_file)


@contextlib.contextmanager
def stdout_to_tqdm():
    save_stdout = sys.stdout
    try:
        sys.stdout = TqdmFile(sys.stdout)
        yield save_stdout
    except Exception as exc:
        raise exc
    finally:
        sys.stdout = save_stdout


if __name__ == '__main__':
    with stdout_to_tqdm() as save_stdout:
        for i in tqdm(range(100), file=save_stdout, ncols=10):
            time.sleep(0.1)
            print("{}".format(i))


