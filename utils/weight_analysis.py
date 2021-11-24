import numpy as np
import torch


def weight_analyser(checkpoint="../weight/0.816_mPCK_1129epoch_myhandnet1.pt"):
    state = torch.load(checkpoint)["model_state"]
    keys = state.keys()
    for key in keys:  # 基本上都是 -0.1~+0.1
        if not 'conv' in key:
            continue
        print("-" * 100)
        print(f"{key=}".center(20, '-'))
        a = state[key].type(torch.float32)
        print(f"{a.mean()=}")
        print(f"{a.std()=}")
        print(f"{a.max()=}")
        print(f"{a.min()=}")


if __name__ == '__main__':
    cp = "../weight/0.104_mPCK_handnet.pt"
    weight_analyser()
