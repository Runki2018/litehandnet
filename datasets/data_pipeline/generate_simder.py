import numpy as np

class GenerateSimDR:
    def __init__(self, sigma=2, k=2):
        super().__init__()
        self.sigma = sigma
        self.k = k   # simdr_split_ratio
 
    def _generate_sa_simdr(self, joints, target_weight, image_size):
        """
        :param joints:  [num_joints, 3]
        :param target_weight: [num_joints, 1] (1: visible, 0: invisible)
        :param image_size: tuple or list
        :return: target
        """

        num_joints = joints.shape[0]
        target_x = np.zeros((num_joints, int(self.image_size[0] * self.k)),
                            dtype=np.float32)
        target_y = np.zeros((num_joints, int(self.image_size[1] * self.k)),
                            dtype=np.float32)

        for joint_id in range(num_joints):
            if target_weight[joint_id] > 0:
                mu_x, mu_y = joints[joint_id, :2] * self.k

                x = np.arange(0, int(image_size[0] * self.k), 1, np.float32)
                y = np.arange(0, int(image_size[1] * self.k), 1, np.float32)

                target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * self.sigma ** 2)))
                target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * self.sigma ** 2)))

        return target_x, target_y

    def __call__(self, results):
        joints = results['joints_3d']
        joints_vis = results['joints_3d_visible']
        image_size = results['ann_info']['image_size']

        target_x, target_y =  self._generate_sa_simdr(joints, joints_vis, image_size)
        results['simdr_x'] = target_x
        results['simdr_y'] = target_y
        return results



