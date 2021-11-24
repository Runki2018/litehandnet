
import os
import sys
sys.path.insert(0, os.path.abspath('.' + '/'))
print(sys.path)
from data import ZHhand_crop_loader_rt
from loss.heatmapLoss import TotalLoss
from models.posenet import SRHandNet

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

if __name__ == '__main__':
    # define global variables
    cwd = os.path.abspath('.')
    root = cwd + "/test/test_example/"
    file = cwd + "/test/test_example/two_samples.json"

    model = SRHandNet(in_dim=256, out_dim=25)
    criterion = TotalLoss()
    dataset, test_loader = ZHhand_crop_loader_rt(batch_size=1).train(img_root=root, ann_file=file)

    for img, target, target_weight, label, bbox in test_loader:
        print(f"{img.shape}")
        print(f"{bbox=}")

        # get diverse heatmaps
        last_hm = target[-1]
        print(f"{last_hm.shape=}")
        region_maps = last_hm[:, :3]
        keypoints_maps = last_hm[:, 3:]  # 1 background + 21 keypoints

        # calculate loss
        # pred_output = deepcopy(target)
        # pred_output[0] = pred_output[0] + 3

        pred_output = model(img)
        print(f"{len(pred_output)=}")
        for pred in pred_output:
            print(f"{pred.shape=}")
        for t in target:
            print(f"{t.shape=}")

        loss = criterion(pred_output, target, target_weight, 5)
        print(f"{loss=}")
