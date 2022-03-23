import torch
from torch import nn
from loss.heatmapLoss import RegionLoss, MaskLoss, L2Loss, JointsMSELoss, SmoothL1Loss, KLFocalLoss
from loss.centernet_simdr_loss import KLDiscretLoss
from collections import defaultdict

loss_func = {
    'KL': KLFocalLoss,
    "MaskLoss": MaskLoss,
    "RegionLoss": RegionLoss,
    "MSE": nn.MSELoss,
    'L2Loss': L2Loss,
    'SmoothL1Loss': SmoothL1Loss,
    'JointsLoss': JointsMSELoss,
}

class MultiTaskLoss(nn.Module):
    """
    MTL多任务学习，自动权重调节: https://zhuanlan.zhihu.com/p/367881339
    """
    def __init__(self, cfg):
        super().__init__()
        self.with_region_map = cfg['with_region_map']
        self.with_cycle_detection = cfg['with_cycle_detection']
        self.with_simdr = True if cfg['simdr_split_ratio'] > 0 else False
        
   
        self.hm_factor = cfg['hm_loss_factor'] 
        self.hm_loss = nn.ModuleList(
                [
                    loss_func[cfg["kpt_loss"]]() if factor > 0 else None
                    for factor in cfg['hm_loss_factor']  
                ]
            )   
        self.region_factor = cfg['region_loss_factor'] 
        self.region_loss = nn.ModuleList(
                [
                    loss_func[cfg["region_loss"]]()
                    if self.with_region_map and factor > 0 else None
                    for factor in cfg['region_loss_factor']  
                ]
            )
    
        self.simdr_loss = KLDiscretLoss() if self.with_simdr else None
        
        self.auto_weight = cfg['auto_weight']
        if self.auto_weight:
            params = torch.ones(cfg['num_loss'], requires_grad=True)
            self.p = nn.Parameter(params, requires_grad=True)   # TODO:将这个参数也放入优化器的参数优化列表中
        
    def forward(self, outputs, targets, target_weight, 
                output_x=None, output_y=None, target_x=None, target_y=None):
        self._forward_check(outputs, targets)

        loss_dict = defaultdict(list)
        self._outputs_loss(outputs, targets, target_weight, loss_dict)

        if self.with_simdr:
            loss_dict['simdr'].append(
                self.simdr_loss(output_x, output_y, target_x, target_y, target_weight)
                )
        
        for k, v in loss_dict.items():
            loss_dict[k] = sum(v)
        
        loss_sum = 0    
        if self.auto_weight:
            for idx, loss in enumerate(loss_dict.values()):
                c2 = self.p[idx] ** 2  # 正则项平方非负，后面再用log(1+c2)，使c2接近0
                loss_sum = loss_sum + 0.5 / c2 * loss + torch.log(1 + c2)
        else:
            loss_sum = sum([l for l in loss_dict.values()])
        
        loss_dict = {k:v.item() for k, v in loss_dict.items()}
        return loss_sum, loss_dict
    
    def _outputs_loss(self, outputs, targets, target_weight, loss_dict):   
        if self.with_region_map:
            for idx, (_pred, _gt) in enumerate(zip(outputs, targets)):
                if self.hm_loss[idx]:
                    kpt_loss = self.hm_loss[idx](_pred[:, :-2], _gt[:, :-2], target_weight) 
                    loss_dict['kpt'].append(kpt_loss * self.hm_factor[idx])
                    
                if self.region_loss[idx]:
                    wh_loss = self.region_loss[idx](_pred[:, -2:], _gt[:, -2:])     
                    loss_dict['wh'].append(wh_loss * self.region_factor[idx])
        else:
            for idx, (_pred, _gt) in enumerate(zip(outputs, targets)):
                if self.hm_loss[idx]:
                    kpt_loss = self.hm_loss[idx](_pred, _gt, target_weight)
                    loss_dict['kpt'].append(kpt_loss * self.hm_factor[idx])
    
    def _forward_check(self, outputs, targets):
        def _check(a, b):
            assert isinstance(a, (tuple, list, nn.ModuleList)), \
                "want a tuple or list, but get {}!!!".format(type(a))
            assert isinstance(b, (tuple, list, nn.ModuleList)), \
                "want a tuple or list, but get {}!!!".format(type(b))
            assert len(a) == len(b), \
                "The length is not equal !!!, {} <> {}".format(len(a), len(b))

        _check(outputs, targets)
        _check(outputs, self.hm_loss)
        _check(self.hm_loss, self.hm_factor)
        
        if self.with_region_map:
            _check(outputs, self.region_loss)
            _check(self.region_factor, self.region_loss)
        
     
        

