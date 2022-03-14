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

from utils.training_kits import set_seeds
from config import seed, pcfg, DATASET, config_dict as cfg

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)
# set_seeds(seed)

class CS_HandData(Dataset):
    """
        同于读取freihand和zhhand等单手数据集，而且不考虑手部的可见性，关键点全都视为可见，因此去除了target_weight,
        先对于zhhand100， PCK已经达到0.976，但是mask和region map的预测有点问题，
        目前改进的想法：region map中的中心点热图，与手部21个关键点一起预测。
        然后 22个点的热图和mask热图在深度方向上拼接在一起，用于预测region map 中的宽和高热图。
        mask背景热图上对于像手腕点那样的，远离其他点的离散点预测效果较差，而且密集区域值越大，这个对后面做乘积不太好，应该都是1才对。
        通道顺序： 0~1，宽高热图，2：背景热图， 3~24：中心点热图和21个关键点热图
    """

    def __init__(self, img_root, ann_file, is_train=False):

        super(CS_HandData, self).__init__()
        self.img_root = img_root
        if os.path.splitext(ann_file)[-1] != ".json":
            raise ValueError("{} is not a json file!".format(ann_file))
        self.ann_json = json.load(open(ann_file, 'r'))
  
        self.img_list = self.ann_json["images"]
        self.ann_list = self.ann_json["annotations"]
 
        self.n_joints = cfg['n_joints']
        self.simdr_split_ratio = cfg['simdr_split_ratio']
        self.max_num_object = pcfg["max_num_bbox"]  # 一张图片上最多保留的目标数
        
        # mask_func = {1: get_mask1, 2: get_mask2, 3: get_mask3}
        # self.get_mask = mask_func[cfg["mask_type"]]  # 手部区域部分，要用背景热图，还是手部关键点掩膜

        # 数据增广的概率
        self.gamma_prob = cfg['gamma_prob']
        self.sigmoid_prob = cfg['sigmoid_prob']
        self.homography_prob = cfg['homography_prob']
        self.flip_prob = cfg['flip_prob']

        self.color_rgb = True
        self.is_train = is_train
        self.cycle_detection_reduction = cfg['cycle_detection_reduction']
        self.hm_size = cfg["hm_size"]  # 输出热图的分辨率
        self.hm_sigma = cfg["hm_sigma"]  # 输出热图的有效范围
        self.bbox_alpha = cfg['bbox_alpha']  # 关键点边框放大倍率

        self.is_augmentation = cfg["is_augment"]  # 是否采用数据增广
        self.image_size = cfg["image_size"]  # 将原图先统一放缩化到新分辨率
        # todo 检查图像标准化的作用
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 改变维度含义： （h,w,c） -> (c, h, w)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # todo 检查关键点权重的影响程度， 在这里其实没用，只是代码当作一个函数的输入参数而保留
        self.use_different_joints_weight = cfg['use_different_joints_weight']
        self.joints_weight = np.asarray(
            [1.1,
             1., 1., 1.1, 1.1,
             1., 1., 1.1, 1.1,
             1., 1., 1., 1.,
             1., 1., 1., 1.,
             1., 1., 1.1, 1.1], dtype=np.float32
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
        # todo 多手数据集仍待解决
        keypoints = np.array(annotation["keypoints"]).reshape((1, self.n_joints, 3))  # (n_hand, n_joints, xyc), 单手数据集n_hand=1

        # 读入图像
        image = cv2.imread(self.img_root + file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            raise ValueError(' Fail to read %s' % image)

        # 缩放图像和关键点到指定的大小
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_NEAREST)
        scale_ratio = np.array(self.image_size) / np.array(original_size)
        keypoints[..., :2] *= scale_ratio
        # bbox = annotation['bbox'] # [cx, cy, w, h]
        # bbox[0], bbox[1] = max(0, (bbox[0] + bbox[2]) / 2), max(0, (bbox[1] + bbox[3]) / 2)
        # bbox = np.array([bbox])
        # bbox[:, :2] *= scale_ratio        
        # bbox[:, 2:] *= scale_ratio       
         
        bbox = get_bbox(keypoints, alpha=self.bbox_alpha)  # [cx, cy, w, h]
        # print(f"2\t{bbox / bbox_alpha}")        
        # 数据增广:
        if self.is_train and self.is_augmentation: 
            image = adjust_gamma(image, prob=self.gamma_prob)
            image = adjust_sigmoid(image, prob=self.sigmoid_prob)
            image, keypoints = homography(image, keypoints, prob=self.homography_prob)
            image, keypoints = horizontal_flip(image, keypoints, prob=self.flip_prob)
            keypoints = validate_keypoints(keypoints, img_size=self.image_size)  # 验证关键点是否可见，不可见设置为-1
       
        # 热图权重，可见为1， 不可见（超出图像边界）为0
        target_weight = np.zeros((self.n_joints, 1), dtype=np.int16)
        vis = keypoints[0, :, 2] > 0
        target_weight[vis] = 1
        
        # 获取最终的GT:     
        # centermap, centermask, sigmas = self.generate_centermap(bbox)  
        hm_size = self.hm_size[0]
        kpts_hm = self.get_heatmaps(keypoints, hm_size, self.hm_sigma[0], bbox)
        
        target_x, target_y = self.generate_sa_simdr(keypoints[0], target_weight, self.hm_sigma[0])
        image = image.astype(np.float32) / 255.0  # todo 在张量化和标准化前，先归一化
        image = self.transform(image)
        # label = 0 if validate_label(keypoints, self.image_size) else label  # 验证手部区域是否有多个点超出图像边缘
        
        # todo: 多守数据集需要完善这个代码
        gt_kpts = np.zeros((self.max_num_object, self.n_joints, 3))
        gt_kpts[0] = keypoints[0]        

        # return image, target_x, target_y, target_weight, centermap, centermask, bbox, gt_kpts
        return image, target_x, target_y, target_weight, kpts_hm, bbox, gt_kpts
    
    def generate_cd_gt(self, img, gt_kpts, bbox, target_weight=None):
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
        # bbox_list = []
        target_x_list = []
        target_y_list = []
        kpts_hm_list = []    

        hm_size = self.hm_size[0] // self.cycle_detection_reduction
        for i in range(batch):
            scale_factor = np.random.rand(1) / 2 + 1
            x, y, w, h = bbox[i][0][:4]
            x, y, w, h = np.array([x, y, w, h]) * scale_factor
            x1, y1 = int(x - w/2 + 0.5), int(y - h/2 + 0.5)
            x2, y2 = int(x + w/2 + 0.5), int(y + h/2 + 0.5)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.image_size[0], x2), min(self.image_size[0], y2)
            w, h = x2 - x1, y2 - y1
            
            img_crop = img[i:i+1, :, y1:y2, x1:x2]  # (1, 3, w, h)

            bbox_crop = np.array([[size_w // 2, size_h // 2, size_w, size_h]])
            
            # mode 默认为nearest， 其他modes: linear | bilinear | bicubic | trilinear
            img_crop = torch.nn.functional.interpolate(img_crop, size=(size_w, size_h)) 
            # img_crop = torch.nn.functional.interpolate(img_crop, size=(nh, nw), mode='linear', align_corners=True) 
  
            # 裁剪图下的关键点坐标
            gt_kpts[i, 0] -= torch.tensor([x1, y1, 0]).to(current_device)
            gt_kpts[i, 0] *= torch.tensor([w / size_w, h / size_h, 1]).to(current_device)
            
            kpts_hm = self.get_heatmaps(gt_kpts[i, 0:1], hm_size, self.hm_sigma[0], bbox_crop, size_h)
            joints = gt_kpts[i, 0].clone().cpu().numpy()
            tw = target_weight[i].clone().cpu().numpy()  
            target_x, target_y = self.generate_sa_simdr(joints, tw, self.hm_sigma[0])
            
            img_crop_list.append(img_crop)
            # bbox_list.append(bbox_crop.tolist())       
            target_x_list.append(torch.as_tensor(target_x))
            target_y_list.append(torch.as_tensor(target_y))
            kpts_hm_list.append(torch.as_tensor(kpts_hm))

        img_crop = torch.cat(img_crop_list, dim=0).to(current_device)
        target_x = torch.stack(target_x_list, dim=0).to(current_device)
        target_y = torch.stack(target_y_list, dim=0).to(current_device)
        kpts_hm = torch.stack(kpts_hm_list, dim=0).to(current_device)
        # bbox_list = torch.as_tensor(bbox_list)
        
        return img_crop, target_x, target_y, kpts_hm
        # return img_crop, target_x, target_y, kpts_hm, bbox_list, gt_kpts
        
    def generate_centermap(self, bbox):
        """ 生成CenterNet中真值热图

        Args:
            bbox (np.array): [n_hand, 4] (cx, cy, w. h)

        Returns:
            target_map, mask: mask 是用于计算宽高和偏置热图的损失时用到
        """
        
        batch_hm = np.zeros((1, self.hm_size[1], self.hm_size[0]), dtype=np.float32)
        batch_wh = np.zeros((2, self.hm_size[1], self.hm_size[0]), dtype=np.float32)
        batch_reg = np.zeros((2, self.hm_size[1], self.hm_size[0]), dtype=np.float32)
        batch_reg_mask = np.zeros((1, self.hm_size[1], self.hm_size[0]), dtype=np.float32)
        scale_factor = np.array(self.image_size) / np.array(self.hm_size) 
        
        n_bbox = bbox.shape[0]
        sigmas = []
        for i in range(n_bbox):
            w, h = bbox[i, 2:]
            if h > 0 and w > 0:
                # 根据bbox大小确定点可移动的半径，保证IOU>thr， 原理可以看CornerNet
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))  
                sigmas.append(radius / 3)  # todo 这个参数有问题，sigma = 8~14
                
                # 中心点 和 整型中心点
                ct = bbox[i, :2] / scale_factor
                ct_int = ct.astype(np.int32)
                
                # 绘制真值热图
                batch_hm[0] = draw_gaussian(batch_hm[0], ct_int, radius)
                batch_wh[:, ct_int[1], ct_int[0]] = w / self.image_size[0], h / self.image_size[1]
                batch_reg[:, ct_int[1], ct_int[0]] = ct - ct_int
                batch_reg_mask[0, ct_int[1], ct_int[0]] = 1
        return np.concatenate([batch_hm, batch_wh, batch_reg], axis=0), batch_reg_mask, sigmas
                
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

                # target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))
                # target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))
                
                target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * sigma ** 2))) 
                target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * sigma ** 2)))
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target_x, target_y
    
    def get_heatmaps(self, keypoints, hm_size, hm_sigma=None, bbox=None, img_size=None):
        # 热图各个通道的含义 heatmaps =  3 region map  [ + 1 background] + 21 keypoints
        region_map, _, = get_multi_regions(bbox, hm_size, hm_sigma, img_size)

        # target heatmap shape = (n_hand, n_joints, h_hm, w_hm)
        kpts_hm, _ = generate_multi_heatmaps(keypoints[..., :2], hm_size, self.joints_weight, bbox, hm_sigma, img_size)  # heatmap

        # TODO： 这个地方在多手的时候需要修改一下，直接叠加是会出问题的, 自己设计的combine_together函数来设计这个真值
        region_map = combine_together(region_map)  # 将多只手的区域图叠加， (n_hand, 3, wh,wh) -> (3, wh,wh)
        kpts_hm = combine_together(kpts_hm)  # 将多只手，同一个关键点的的热图叠加, (n_hand, 21, wh,wh) -> (21, wh,wh)

        return np.concatenate((region_map, kpts_hm), axis=0)


class CS_Loader:
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
        dataset = CS_HandData(img_root=self.root, ann_file=self.test_file, is_train=False)
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
        dataset = CS_HandData(img_root=self.root, ann_file=self.train_file, is_train=True)
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
