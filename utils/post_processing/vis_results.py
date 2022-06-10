import math
import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path

class SaveResultImages:
    def __init__(self, dataset, output_path):
        self.pose_link_color = dataset.pose_link_color  # [(r, g, b), ...]
        self.pose_kpt_color = dataset.pose_kpt_color    # [(r, g, b), ...]
        self.pose_skeleton = dataset.pose_skeleton      # [[0, 1], [], ...]
        self.output_path = Path(output_path)
    
    
    def _add_joints(self, image, joints):
        """
            在图片上画出关键点和骨架连线
        Args:
            image (np.ndarray): [H, W, 3]
            joints (np.ndarray):[K, 3]
        """
        num_joints = joints.shape[0]
        
        def link(a, b, color):
            if a < num_joints and b < num_joints:
                jointa = joints[a]
                jointb = joints[b]
                if jointa[2] > 0 and jointb[2] > 0:
                    cv2.line(
                        image,
                        (int(jointa[0]), int(jointa[1])),
                        (int(jointb[0]), int(jointb[1])),
                        color,
                        2
                    )
        # add joints
        for i in range(num_joints):
            if joints[i, 2] > 0:
                cv2.circle(image, (int(joints[i, 0]), int(joints[i, 1])), 
                           1, self.pose_kpt_color[i], 2)
        
        # add link
        for i, pair in enumerate(self.pose_skeleton):
            link(pair[0], pair[1], self.pose_link_color[i])

        return image
    
    def _add_bboxes(self, image, bbox, color):
        """在图片上画出边界框

        Args:
            image (np.ndarray): [H, W, 3]
            bbox (np.ndarray): [x1, y1, x2, y2]
            color (np.ndarray): [r, g, b]

        Returns:
            image: [H, W, 3]
        """
        H, W = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        leftTop = (int(x1), int(y1))  # 左上角的点坐标 (x,y)
        rightBottom = (int(x2), int(y2))  # 右下角的点坐标= (x+w,y+h)
        image = cv2.rectangle(image, leftTop, rightBottom, color, 4, 8)
        return image


    def save_images_with_joints(self, batch_image, batch_joints, batch_joints_vis,
                                     file_name, nrow=8, padding=2):
        '''
            batch_image: [batch_size, channel, height, width]
            batch_joints: [batch_size, num_joints, 3],
            batch_joints_vis: [batch_size, num_joints, 1]
            file_name: eg. 'epoch1.jpg'
        '''
        grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)

        nmaps = batch_image.size(0)
        xmaps = min(nrow, nmaps)
        ymaps = int(math.ceil(float(nmaps) / xmaps))
        height = int(batch_image.size(2) + padding)
        width = int(batch_image.size(3) + padding)
        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                joints = batch_joints[k]
                joints_vis = batch_joints_vis[k]

                for joint, joint_vis in zip(joints, joints_vis):
                    joint[0] = x * width + padding + joint[0]
                    joint[1] = y * height + padding + joint[1]
                    if joint_vis[0]:
                        cv2.circle(
                            ndarr,
                            (int(joint[0]), int(joint[1])),
                            2,
                            [255, 0, 0],
                            2
                        )
                k = k + 1
        cv2.imwrite(str(self.output_path.joinpath(file_name)), ndarr)
        
    def save_images_with_heatmap(self, batch_image, batch_heatmap, file_name, padding=2):
        '''
            image(tensor): [Batch, 3, height, width]
            heatmap(tensor): [batch, num_joints, 3]
            file_name: eg. 'heatmap.jpg'
            padding: 每张图片填充边缘距离,便于区分图像边界。
        '''
        batch_heatmap = batch_heatmap.mul(255).clamp(0, 255).byte().cpu().detach().numpy()

        batch_size, num_joints, hm_height, hm_width = batch_heatmap.shape
        # 将图片下采样到热图大小
        batch_image = torch.nn.functional.interpolate(batch_image, size=(hm_height, hm_width))
        nrow = 1 + num_joints  # grid网格中每一行有多少张图片

        images = batch_image.unsqueeze(dim=1)
        images = torch.tile(images, (1, nrow, 1, 1, 1))  
        images = images.reshape((-1, 3, hm_height, hm_width))
        # [batch_size, nrow, 3, height, width]
        grid = torchvision.utils.make_grid(images, nrow, padding, True)
        # [batch_size*height, nrow*width, 3]
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
        ndarr = ndarr.astype(np.float32)

        # grid中一个网格的宽高
        height = int(hm_height + padding)
        width = int(hm_width + padding)
        half_padding = padding // 2

        for i in range(batch_size):
            for j in range(1, nrow):
                heatmap = batch_heatmap[i, j-1]
                colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                h1, h2 = height * i + half_padding, height * (i+1) - half_padding
                w1, w2 = width * j + half_padding, width * (j+1) - half_padding
                ndarr[h1:h2, w1:w2, :] *= 0.3 # 保留30%的原图作为底色
                ndarr[h1:h2, w1:w2, :] += 0.7 * colored_heatmap

        cv2.imwrite(str(self.output_path.joinpath(file_name)), ndarr.astype(np.uint8))

