import os
import cv2
import json
from torch.utils.data import DataLoader, Dataset
from data.handset.dataset_function import *
from utils.data_augmentation import adjust_gamma, adjust_sigmoid, homography, horizontal_flip
from config.config import config_dict


class ZHhandCrop(Dataset):
    """
        the samples are interception of ZHhand images, and all the samples just include
        one hand.
        revert keypoints heatmap
    """

    def __init__(self, img_root, ann_file,
                 gamma_prob=0.5,
                 sigmoid_prob=0.5,
                 homography_prob=0.5,
                 flip_prob=0.5,
                 n_joints=21,
                 color_rgb=True):

        super(ZHhandCrop, self).__init__()
        self.img_root = img_root
        if os.path.splitext(ann_file)[-1] != ".json":
            raise ValueError("{} is not a json file!".format(ann_file))
        self.ann_json = json.load(open(ann_file, 'r'))
        self.img_list = self.ann_json["images"]
        self.ann_list = self.ann_json["annotations"]
        self.n_joints = n_joints

        # 数据增广的概率
        self.gamma_prob = gamma_prob
        self.sigmoid_prob = sigmoid_prob
        self.homography_prob = homography_prob
        self.flip_prob = flip_prob

        self.color_rgb = color_rgb
        self.hm_size = config_dict["hm_size"]  # 输出热图的分辨率
        self.hm_sigma = config_dict["hm_sigma"]  # 输出热图的有效范围
        self.n_supervision = len(self.hm_size)  # 受监督的热图层数 = 中间监督 + 最终目标
        assert len(self.hm_size) == len(self.hm_sigma), \
            "len(hm_size) != len(hm_sigma), {} != {}".format(len(self.hm_size), len(self.hm_sigma))

        self.is_augmentation = config_dict["is_augment"]  # 是否采用数据增广
        self.is_mask = config_dict["is_mask"]  # 手部区域部分，要用背景热图，还是手部关键点掩膜
        self.new_size = config_dict["new_size"]  # 将原图先统一放缩化到新分辨率
        # todo 检查图像标准化的作用
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 改变维度含义： （h,w,c） -> (c, h, w)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # todo 检查关键点权重的影响程度
        self.joints_weight = np.asarray(
            [1.2, 1., 1., 1.2, 1.2,
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
        size = img_info["width"], img_info["height"]
        label = annotation["category_id"]
        keypoints = np.array(annotation["keypoints"]).reshape((1, 21, 3))  # (n_hand, n_joints, xyc), 单手数据集n_hand=1

        # 读入图像
        image = cv2.imread(self.img_root + file_name)

        if self.color_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise ValueError(' Fail to read %s' % image)

        # 缩放图像和关键点到指定的大小
        image = cv2.resize(image, self.new_size, interpolation=cv2.INTER_NEAREST)
        scale_ratio = np.array(self.new_size) / np.array(size)
        keypoints[..., :2] *= scale_ratio

        # 数据增广:
        if self.is_augmentation:
            image = adjust_gamma(image, prob=self.gamma_prob)
            image = adjust_sigmoid(image, prob=self.sigmoid_prob)
            image, keypoints = homography(image, keypoints, prob=self.homography_prob)
            image, keypoints = horizontal_flip(image, keypoints, prob=self.flip_prob)

        # 获取最终的GT:
        hm_output, hm_weight, bbox = self.get_heatmaps(keypoints)

        image = image.astype(np.float32) / 255.0  # todo 在张量化和标准化前，先归一化
        image = self.transform(image)
        label = 0 if validate_label(keypoints, size) else label  # 验证手部区域是否有多个点超出图像边缘

        return image, hm_output, hm_weight, label, bbox

    def get_heatmaps(self, keypoints):
        hm_list = []  # 热图列表，每个列表元素为一个监督的真值热图
        weight_list = []  # 手部21个关键点的权重列表，每个列表元素为一个监督的真值热图
        bbox = None  # todo 只需要最后一个, [c_x, c_y, w, h] ?

        for i in range(self.n_supervision):
            # 热图各个通道的含义 heatmaps = 3 region map + 1 background + 21 keypoints
            hm_output = np.zeros((4 + self.n_joints, self.hm_size[i], self.hm_size[i]), dtype=np.float32)
            hm_weight = np.ones((4 + self.n_joints, 1), dtype=np.float32)

            # target heatmap shape = (n_hand, n_joints, h_hm, w_hm)
            kpts_hm, kpts_weight = generate_multi_heatmaps(keypoints[..., :2],
                                                           self.hm_size[i], self.hm_sigma[i],
                                                           self.joints_weight)  # heatmap
            # todo 超参数 alpha 直接设为1.3，后面可以增加实验寻找合适大小
            region_map, region_weight, bbox = get_multi_regions(keypoints, self.hm_size[i],
                                                                self.hm_sigma[i], alpha=1.3)

            # todo： 这个地方在多手的时候需要修改一下，直接叠加是会出问题的, 自己设计的combine_together函数来设计这个真值
            hm_output[:3] = combine_together(region_map)  # 将多只手的区域图叠加
            hm_output[4:] = combine_together(kpts_hm)    # 将多只手，同一个关键点的的热图叠加
            hm_output[3] = get_mask2(hm_output[4:]) if self.is_mask else get_mask1(hm_output[4:])

            # get a ground truth
            hm_list.append(hm_output)
            weight_list.append(hm_weight)

        return hm_list, weight_list, bbox[0]


class ZHhandCropLoader:
    def __init__(self, batch_size=16, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def test(self, img_root, ann_file):
        """

        :return: the data loader of testing dataset
        """
        dataset = ZHhandCrop(img_root=img_root, ann_file=ann_file)
        print("sample number of testing dataset: ", dataset.__len__())
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers,
                            drop_last=True)
        return dataset, loader

    def train(self, img_root, ann_file):
        """
        :return dataset and the data loader of training dataset
        """
        dataset = ZHhandCrop(img_root=img_root, ann_file=ann_file,
                             gamma_prob=0.5, sigmoid_prob=0.5,
                             homography_prob=0.5, flip_prob=0.5)

        print("sample number of training dataset: ", dataset.__len__())
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers,
                            drop_last=True)
        return dataset, loader


if __name__ == '__main__':
    # image_path = "../image/00000000.jpg"
    image_path = "D:/python/Project/labelKPs/LabelKPs/sample/data/0/img_0/6_810_rest_810_rgb-1560461435362346_60.jpg"
    # img = cv2.imread(image_path)
    from torchvision import transforms
    from utils.visualization_tools import draw_heatmaps

    img = np.array([[0, 1, 255], [255, 1., 25], [4, 127, 128]], dtype=np.float32)
    img = np.stack((img, img, img)) / 255
    print(img)
    transform = transforms.Compose([
        transforms.ToTensor(),  #
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    print(img)
    print(img.shape)
    draw_heatmaps(img.unsqueeze(dim=0), title="image", k=3)
    # print(img[50:150, 50:150])
    # print(np.max(img))
