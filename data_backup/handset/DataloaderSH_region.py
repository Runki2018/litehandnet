import os
import cv2
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from data.handset.dataset_function import *
from utils.data_augmentation import adjust_gamma, adjust_sigmoid, homography, horizontal_flip

from utils.training_kits import set_seeds
from config.config import seed, DATASET, pcfg, config_dict as cfg

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)
# set_seeds(seed)


class HandData(Dataset):
    """
        同于读取freihand和zhhand等单手数据集，而且不考虑手部的可见性，关键点全都视为可见，因此去除了target_weight,
        先对于zhhand100， PCK已经达到0.976，但是mask和region map的预测有点问题，
        目前改进的想法：region map中的中心点热图，与手部21个关键点一起预测。
        然后 22个点的热图和mask热图在深度方向上拼接在一起，用于预测region map 中的宽和高热图。
        mask背景热图上对于像手腕点那样的，远离其他点的离散点预测效果较差，而且密集区域值越大，这个对后面做乘积不太好，应该都是1才对。
        通道顺序： 0~1，宽高热图，2：背景热图， 3~24：中心点热图和21个关键点热图
    """

    def __init__(self, img_root, ann_file, is_train=False):

        super(HandData, self).__init__()
        self.img_root = img_root
        if os.path.splitext(ann_file)[-1] != ".json":
            raise ValueError("{} is not a json file!".format(ann_file))
        self.ann_json = json.load(open(ann_file, 'r'))
        self.img_list = self.ann_json["images"]
        self.ann_list = self.ann_json["annotations"]
        self.n_joints = cfg['n_joints']

        # 数据增广的概率
        self.gamma_prob = cfg['gamma_prob']
        self.sigmoid_prob = cfg['sigmoid_prob']
        self.homography_prob = cfg['homography_prob']
        self.flip_prob = cfg['flip_prob']

        self.color_rgb = True
        self.is_train = is_train
        self.hm_size = cfg["hm_size"]  # 输出热图的分辨率
        self.hm_sigma = cfg["hm_sigma"]  # 输出热图的有效范围

        self.is_augmentation = cfg["is_augment"]  # 是否采用数据增广
        mask_func = {1: get_mask1, 2: get_mask2, 3: get_mask3}
        self.get_mask = mask_func[cfg["mask_type"]]  # 手部区域部分，要用背景热图，还是手部关键点掩膜
        self.image_size = cfg["image_size"]  # 将原图先统一放缩化到新分辨率
        # todo 检查图像标准化的作用
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 改变维度含义： （h,w,c） -> (c, h, w)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # todo 检查关键点权重的影响程度， 在这里其实没用，只是代码当作一个函数的输入参数而保留
        self.joints_weight = np.asarray(
            [1.2,
             1., 1., 1.2, 1.2,
             1., 1., 1.2, 1.2,
             1., 1., 1., 1.,
             1., 1., 1., 1.,
             1., 1., 1.2, 1.2], dtype=np.float32
        ).reshape((self.n_joints, 1))

    def __len__(self):
        return len(self.ann_list)

    def __getitem__(self, idx):
        img_info = self.img_list[idx]
        annotation = self.ann_list[idx]

        # 读标注信息
        file_name = img_info["file_name"].strip("./")  # todo 去除路径中的 ./
        original_size = img_info["width"], img_info["height"]
        label = annotation["category_id"]
        keypoints = np.array(annotation["keypoints"]).reshape((1, 21, 3))  # (n_hand, n_joints, xyc), 单手数据集n_hand=1
        self.simdr_split_ratio = cfg['simdr_split_ratio']
        self.max_num_object = pcfg["max_num_bbox"]  # 一张图片上最多保留的目标数

        # 读入图像
        image = cv2.imread(self.img_root + file_name)

        if self.color_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise ValueError(' Fail to read %s' % image)

        # 缩放图像和关键点到指定的大小
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_NEAREST)
        scale_ratio = np.array(self.image_size) / np.array(original_size)
        keypoints[..., :2] *= scale_ratio

        # 数据增广:
        if self.is_train and self.is_augmentation:
            image = adjust_gamma(image, prob=self.gamma_prob)
            image = adjust_sigmoid(image, prob=self.sigmoid_prob)
            image, keypoints = homography(image, keypoints, prob=self.homography_prob)
            image, keypoints = horizontal_flip(image, keypoints, prob=self.flip_prob)
            keypoints = validate_keypoints(keypoints, img_size=(256, 256))
        
        # 热图权重，可见为1， 不可见（超出图像边界）为0
        hm_weight = np.zeros((self.n_joints, 1), dtype=np.int16)
        vis = keypoints[0, :, 2] > 0
        hm_weight[vis] = 1
        if cfg['gt_mode']['region_map']:
            hm_weight = np.concatenate((np.ones((3, 1)), hm_weight), axis=0)  # add region weight
        
        bbox = get_bbox(keypoints, alpha=1.3)
            
        # 获取最终的GT:
        hm_target = []
        pre_param = [0, 0]  # do not repeat again if param is not changed
        for hm_size, hm_sigma in zip(cfg['hm_size'], cfg['hm_sigma']):
            if hm_size == pre_param[0] and hm_sigma == pre_param[1]:
                hm_target.append(hm_target[-1])
                continue 
            else:
                pre_param = [hm_size, hm_sigma]
                if cfg['gt_mode']['region_map']:           
                    hm = self.get_heatmaps(keypoints, bbox, hm_size, hm_sigma)
                              
                elif cfg['gt_mode']['just_kpts']:
                    hm, _ = generate_multi_heatmaps(keypoints[..., :2], hm_size, self.joints_weight, bbox, heatmap_sigma=hm_sigma)  # heatmap
                    hm = combine_together(hm)  # 将多只手，同一个关键点的的热图叠加, (n_hand, 21, wh,wh) -> (21, wh,wh)
                hm_target.append(hm)      

        image = image.astype(np.float32) / 255.0  # todo 在张量化和标准化前，先归一化
        image = self.transform(image)
        # 验证手部区域是否有多个点超出图像边缘
        label = 0 if validate_label(keypoints, self.image_size) else label  

        return image, hm_target, hm_weight, label, bbox
       
    def get_heatmaps(self, keypoints, bbox, hm_size, hm_sigma=None):
        # 热图各个通道的含义 heatmaps =  3 region map  [ + 1 background] + 21 keypoints
        # todo 超参数 alpha 直接设为1.3，后面可以增加实验寻找合适大小
        region_map, _, = get_multi_regions(bbox, hm_size, hm_sigma)

        # target heatmap shape = (n_hand, n_joints, h_hm, w_hm)
        kpts_hm, _ = generate_multi_heatmaps(keypoints[..., :2], hm_size, self.joints_weight, bbox, heatmap_sigma=hm_sigma)  # heatmap

        # todo： 这个地方在多手的时候需要修改一下，直接叠加是会出问题的, 自己设计的combine_together函数来设计这个真值
        region_map = combine_together(region_map)  # 将多只手的区域图叠加， (n_hand, 3, wh,wh) -> (3, wh,wh)
        kpts_hm = combine_together(kpts_hm)  # 将多只手，同一个关键点的的热图叠加, (n_hand, 21, wh,wh) -> (21, wh,wh)
        # hm_output = np.concatenate((region_map, kpts_hm), axis=0)
        mask = self.get_mask(kpts_hm)

        return np.concatenate((region_map, kpts_hm, mask), axis=0)
    

class RegionLoader:
    def __init__(self):
        self.batch_size = cfg["batch_size"]
        self.num_workers = cfg["workers"]
        self.root = DATASET['root']
        self.test_file = DATASET['test_file']
        self.train_file = DATASET['train_file']

    def test(self, just_dataset=False):
        """
        :return: the data loader of testing dataset
        """
        dataset = HandData(img_root=self.root, ann_file=self.test_file, is_train=False)
        print("sample number of testing dataset: ", dataset.__len__())
        if just_dataset:
            return dataset
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            drop_last=True,
                            pin_memory=True)
        return dataset, loader

    def train(self, just_dataset=False):
        """
        :return dataset and the data loader of training dataset
        """
        dataset = HandData(img_root=self.root, ann_file=self.train_file, is_train=True)
        print("sample number of training dataset: ", dataset.__len__())
        if just_dataset:
            return dataset
            
        loader = DataLoader(dataset=dataset, 
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            drop_last=True,
                            pin_memory=cfg['pin_memory'])
        return dataset, loader


if __name__ == '__main__':
    a = 1
