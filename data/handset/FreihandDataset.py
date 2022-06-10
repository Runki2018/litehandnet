import imp
import os
import cv2
import json
import math
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from data.handset.dataset_function import *
from utils.utils_centermap import *
from utils.data_augmentation import adjust_gamma, adjust_sigmoid, homography, horizontal_flip

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)
from config import pcfg
cfg = None 

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

        self.img_list = self.ann_json["images"][::40]
        self.ann_list = self.ann_json["annotations"][::40]

        self.n_joints = cfg['n_joints']
        self.simdr_split_ratio = cfg['simdr_split_ratio']
        self.with_region_map = cfg['with_region_map']  # 是否使用 region map
        self.max_num_object = pcfg["max_num_bbox"]  # 一张图片上最多保留的目标数
        
        # 数据增广的概率
        self.gamma_prob = cfg['gamma_prob']
        self.sigmoid_prob = cfg['sigmoid_prob']
        self.homography_prob = cfg['homography_prob']
        self.flip_prob = cfg['flip_prob']

        self.color_rgb = True
        self.cycle_detection_reduction = cfg['cycle_detection_reduction']
        self.hm_size = cfg["hm_size"]  # 输出热图的分辨率
        self.hm_sigma = cfg["hm_sigma"]  # 输出热图的有效范围
        self.bbox_alpha = cfg['bbox_alpha']  # 关键点边框放大倍率
        self.bbox_factor = pcfg['bbox_factor']  # 循环训练时，裁剪框放大倍数

        if cfg['dataset'] != "freihand_plus" and is_train:
            self.is_augmentation = cfg["is_augment"]   # 是否采用数据增广
        else:
            self.is_augmentation = False
        self.image_size = cfg["image_size"]  # 将原图先统一放缩化到新分辨率

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 改变维度含义： （h,w,c） -> (c, h, w)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # todo 检查关键点权重的影响程度， 在这里其实没用，只是代码当作一个函数的输入参数而保留
        # self.joints_weight = np.asarray(
        #     [1.1,
        #      1., 1., 1.1, 1.1,
        #      1., 1., 1.1, 1.1,
        #      1., 1., 1., 1.,
        #      1., 1., 1., 1.,
        #      1., 1., 1.1, 1.1], dtype=np.float32
        # ).reshape((self.n_joints, 1))
        self.joints_weight = None

    def __len__(self):
        return len(self.ann_list)

    def __getitem__(self, idx):
        img_info = self.img_list[idx]
        annotation = self.ann_list[idx]

        # 读标注信息
        file_name = img_info["file_name"].strip("./")  # todo 去除路径中的 ./
        original_size = img_info["width"], img_info["height"]
        label = annotation["category_id"]
        # todo 多手数据集仍待解决
        keypoints = np.array(annotation["keypoints"]).reshape((1, self.n_joints, 3))  # (n_hand, n_joints, xyc), 单手数据集n_hand=1

        # 读入图像
        image = cv2.imread(os.path.join(self.img_root, file_name))

        if self.color_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise ValueError(' Fail to read %s' % image)

        # 缩放图像和关键点到指定的大小
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_NEAREST)
        scale_ratio = np.array(self.image_size) / np.array(original_size)
        keypoints[..., :2] *= scale_ratio
        bbox = np.array(annotation['bbox'], dtype=np.float32)  
        bbox[:2] = bbox[:2] + bbox[2:] / 2  # [lx, ly, w, h] -> [cx, cy, w, h]
        bbox = bbox[None] * np.hstack([scale_ratio, scale_ratio])
        # 验证关键点是否可见，不可见设置为-1    
        keypoints = validate_keypoints(keypoints, img_size=self.image_size) 
        
        # data augmentation
        if self.is_augmentation:
            image = adjust_gamma(image, self.gamma_prob)
            image = adjust_sigmoid(image, self.sigmoid_prob)
            image, keypoints, bbox = homography(image, keypoints, 
                                                self.homography_prob,
                                                bbox, False)
            image, keypoints, bbox = horizontal_flip(image, keypoints,
                                                     self.flip_prob,bbox=bbox)
            # 验证关键点是否可见，不可见设置为-1     
            keypoints = validate_keypoints(keypoints, img_size=self.image_size)       
                
        # 获取最终的GT:
        image = image.astype(np.float32) / 255.0  # todo 在张量化和标准化前，先归一化
        image = self.transform(image)
        # label = 0 if validate_label(keypoints, self.image_size) else label  # 验证手部区域是否有多个点超出图像
        
        gt_kpts = np.zeros((self.max_num_object, self.n_joints, 3))
        gt_kpts[0] = keypoints[0]   
        
        targets = []
        for i in range(len(self.hm_size)):
            if i > 0 and \
                self.hm_size[i] == self.hm_size[i-1] and \
                self.hm_sigma[i] == self.hm_sigma[i-1]:
                    # do not repeat again if param is not changed
                    targets.append(targets[-1])
            else:
                kpts_hm, target_weight =\
                    self.get_heatmaps(
                            keypoints=keypoints,
                            hm_size=self.hm_size[i],
                            hm_sigma=self.hm_sigma[i],
                            bbox=bbox,
                            img_size=self.image_size[0]
                    )
                #  (n_hand, n_joints, 1) -> (n_joints, 1)
                target_weight = target_weight[0]
                targets.append(kpts_hm)

        if self.simdr_split_ratio:
            target_x, target_y = self.generate_sa_simdr(
                                    joints=keypoints[0],
                                    target_weight=target_weight,
                                    sigma=self.hm_sigma[0]
                                )
            return image, targets, target_weight, bbox, gt_kpts, target_x, target_y
        else:
            return image, targets, target_weight, bbox, gt_kpts

    def generate_cd_gt(self, img, gt_kpts, bbox):
        """ 生产循环检测的训练样本, 将bbox范围内的目标裁剪出来 
        Args:
            img (tensor): [batch, c, h, w]
            gt_kpts (tensor): [batch, max_num_bbox, n_joints, 3], (x, y, vis)
            bbox (list): [batch, n_bbox, 4]
            target_weight: (tensor) [n_joints, 1]
        """
        
        current_device = img.device
        batch, c_img, h_img, w_img = img.shape
        size_w =  w_img // self.cycle_detection_reduction
        size_h =  h_img // self.cycle_detection_reduction  
        img_crop_list = []
        kpts_hm_list = []
        
        if self.simdr_split_ratio:    
            target_x_list = []
            target_y_list = []

        hm_size = self.hm_size[0] // self.cycle_detection_reduction
        for i in range(batch):
            x, y, w, h = bbox[i][0]
            w, h = w * self.bbox_factor, h * self.bbox_factor
            x1, y1 = max(0, int(x - w/2)),  max(0, int(y - h/2))
            x2, y2 = min(w_img, int(x + w/2)),  min(h_img, int(y + h/2))
            w, h = x2 - x1, y2 - y1

            img_crop = img[i:i+1, :, y1:y2, x1:x2]  # (1, 3, w, h)
            bbox_crop = np.array([[size_w // 2, size_h // 2, size_w, size_h]])
            
            # mode 默认为nearest， 其他modes: linear | bilinear | bicubic | trilinear
            # img_crop = torch.nn.functional.interpolate(img_crop, size=(nh, nw), mode='linear', align_corners=True) 
            img_crop = torch.nn.functional.interpolate(img_crop, size=(size_h, size_w)) 
            img_crop_list.append(img_crop)
  
            # 裁剪图下的关键点坐标
            gt_kpts[i, 0] -= torch.tensor([x1, y1, 0]).to(current_device)
            gt_kpts[i, 0] *= torch.tensor([size_w / w, size_h / h, 1]).to(current_device)
   
            targets, target_weight = self.get_heatmaps(
                gt_kpts[i, 0:1], hm_size, self.hm_sigma[0], bbox_crop, size_h)
            kpts_hm_list.append(torch.as_tensor(targets))
            
            if self.simdr_split_ratio:
                joints = gt_kpts[i, 0].clone().cpu().numpy()
                target_x, target_y = self.generate_sa_simdr(joints, target_weight[0], self.hm_sigma[0]) 
                target_x_list.append(torch.as_tensor(target_x))
                target_y_list.append(torch.as_tensor(target_y))

        img_crop = torch.cat(img_crop_list, dim=0).to(current_device)
        targets = [torch.stack(kpts_hm_list, dim=0).to(current_device)]
        
        if self.simdr_split_ratio:
            target_x = torch.stack(target_x_list, dim=0).to(current_device)
            target_y = torch.stack(target_y_list, dim=0).to(current_device)
            return img_crop, targets, gt_kpts, target_x, target_y
        else:
            return img_crop, targets, gt_kpts

    def generate_sa_simdr(self, joints, target_weight, sigma):
        """
        :param joints:  [num_joints, 3]
        :param target_weight: [num_joints, 1] (1: visible, 0: invisible)
        :param sigma: 1d gaussion sigma
        :return: target
        """
        sigma = sigma if sigma > 2 else 2
        target_x = np.zeros((self.n_joints,
                             int(self.image_size[0] * self.simdr_split_ratio)),
                            dtype=np.float32)
        target_y = np.zeros((self.n_joints,
                             int(self.image_size[1] * self.simdr_split_ratio)),
                            dtype=np.float32)

        for joint_id in range(self.n_joints):
            if target_weight[joint_id] > 0:
                mu_x, mu_y = joints[joint_id, :2] * self.simdr_split_ratio

                x = np.arange(0, int(self.image_size[0] * self.simdr_split_ratio), 1, np.float32)
                y = np.arange(0, int(self.image_size[1] * self.simdr_split_ratio), 1, np.float32)

                target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * sigma ** 2))) 
                target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * sigma ** 2)))

        if self.joints_weight is not None:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target_x, target_y
    
    def get_heatmaps(self, keypoints, hm_size, hm_sigma, bbox, img_size):
        # 热图各个通道的含义 heatmaps = 21 keypoints + 3 region map
        # target heatmap shape = (n_hand, n_joints, h_hm, w_hm)
        kpts_hm, target_weight = generate_multi_heatmaps(keypoints, img_size,
                                                         hm_size, hm_sigma,
                                                         self.joints_weight)  # heatmap
        
        # 将多只手，同一个关键点的的热图叠加, (n_hand, 21, wh,wh) -> (21, wh,wh)
        kpts_hm = combine_together(kpts_hm)  

        if self.with_region_map:
            region_map = get_multi_regions(bbox, img_size, hm_size, hm_sigma)
            region_map = combine_together(region_map)  # 将多只手的区域图叠加， (n_hand, 3, wh,wh) -> (3, wh,wh)
            target = np.concatenate((kpts_hm, region_map), axis=0)
            # add weight for center map:
            target_weight =np.pad(target_weight, ((0, 0), (0, 1), (0, 0)), 'constant', constant_values=1)
        else:
            target = kpts_hm
        return target, target_weight


class FreiHandLoader:
    def __init__(self, config, DATASET):
        global cfg
        cfg = config
        self.batch_size = cfg["batch_size"]
        self.num_workers = cfg["workers"]
        self.root = DATASET['root']
        self.test_file = DATASET['test_file']
        self.train_file = DATASET['train_file']


    def test(self, just_dataset=False):
        """
        :return: the data loader of testing dataset
        """
        dataset = HandData(img_root=self.root, ann_file=self.test_file,
                           is_train=False)
        print("sample number of testing dataset: ", dataset.__len__())
        if just_dataset:
            return dataset
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            drop_last=False,
                            pin_memory=True)
        return dataset, loader

    def train(self, just_dataset=False):
        """
        :return dataset and the data loader of training dataset
        """
        dataset = HandData(img_root=self.root, ann_file=self.train_file,
                           is_train=True)
        print("sample number of training dataset: ", dataset.__len__())
        if just_dataset:
            return dataset
            
        loader = DataLoader(dataset=dataset, 
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            drop_last=False,
                            pin_memory=cfg['pin_memory'])
        return dataset, loader


if __name__ == '__main__':
    a = 1
