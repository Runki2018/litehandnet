import torch
from torch import nn
from loss.heatmapLoss import RegionLoss, MaskLoss, L2Loss, JointsMSELoss, SmoothL1Loss
from loss.centernet_simdr_loss import focal_loss, reg_l1_loss, KLDiscretLoss
from collections import defaultdict

loss_func = {
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
        
        self.losses = {}
        self.losses['hm_factor'] = cfg['hm_loss_factor'] 
        self.losses['hm'] = nn.ModuleList(
                [
                    loss_func[cfg["kpt_loss"]]() if factor > 0 else None
                    for factor in cfg['hm_loss_factor']  
                ]
            )   
        self.losses['region_factor'] = cfg['region_loss_factor'] 
        self.losses['region'] = nn.ModuleList(
                [
                    loss_func[cfg["region_loss"]]()
                    if self.with_region_map and factor > 0 else None
                    for factor in cfg['region_loss_factor']  
                ]
            )
        
        # ! 循环检测的损失函数，是否需要分开多个损失函数实例去计算？
        self.cd_losses = defaultdict(list)
        self.cd_losses['hm_factor'] = cfg['cd_hm_loss_factor'] 
        self.cd_losses['hm'] = nn.ModuleList(
                [
                    loss_func[cfg["kpt_loss"]]() if factor > 0 else None
                    for factor in cfg['cd_hm_loss_factor']  
                ]
            )   
        self.cd_losses['region_factor'] = cfg['cd_region_loss_factor'] 
        self.cd_losses['region'] = nn.ModuleList(
                [
                    loss_func[cfg["region_loss"]]()
                    if self.with_region_map and factor > 0 else None
                    for factor in cfg['cd_region_loss_factor']  
                ]
            )
        
        self.simdr_loss = KLDiscretLoss() if cfg['simdr_split_ratio'] else None
        
        self.auto_weights = cfg['auto_weights']
        if self.auto_weights:
            params = torch.ones(cfg['num_loss'], requires_grad=True)
            self.p = nn.Parameter(params, requires_grad=True)   # TODO:将这个参数也放入优化器的参数优化列表中
        
    def forward(self, outputs, targets, target_weight, cycle_train=False, 
                output_x=None, output_y=None, target_x=None, target_y=None):
        self._forward_check(self, outputs, targets)

        loss_dict = defaultdict(list)
        if cycle_train:
            self._outputs_loss(outputs, targets, target_weight, loss_dict, self.cd_losses)
        else:
            self._outputs_loss(outputs, targets, target_weight, loss_dict, self.losses)

        if self.with_simdr:
            loss_dict['simdr'].append(
                self.simdr_loss(output_x, output_y, target_x, target_y, target_weight)
                )
        
        for k, v in loss_dict.items():
            loss_dict[k] = sum(v)
        
        loss_sum = 0    
        if self.auto_weights:
            for idx, loss_idx in enumerate(loss_dict.values()):
                c2 = self.p[idx] ** 2  # 正则项平方非负，后面再用log(1+c2)，使c2接近0
                loss_sum = loss_sum + 0.5 / c2 * loss_idx + torch.log(1 + c2)
        else:
            loss_sum = sum([l for l in loss_dict.values()])
        
        loss_dict = {k:v.item() for k, v in loss_dict.items()}
        return loss_sum, loss_dict
    
    def _outputs_loss(self, outputs, targets, target_weight, loss_dict, loss_fuc):   
        if self.with_region_map:
            for idx, (_pred, _gt) in enumerate(zip(outputs, targets)):
                if loss_fuc['hm'][idx]:
                    kpt_loss = loss_fuc['hm'][idx](_pred[:, :-2], _gt[:, :-2], target_weight) 
                    loss_dict['kpt'].append(kpt_loss * loss_fuc['hm_factor'][idx])
                    
                if loss_fuc['region'][idx]:
                    wh_loss =loss_fuc['region'][idx](_pred[:, -2:], _gt[:, -2:])     
                    loss_dict['wh'].append(wh_loss * loss_fuc['region_factor'][idx])
        else:
            for idx, (_pred, _gt) in enumerate(zip(outputs, targets)):
                if loss_fuc['hm'][idx]:
                    kpt_loss = loss_fuc['hm'][idx](_pred, _gt, target_weight)
                    loss_dict['kpt'].append(kpt_loss * loss_fuc['hm_factor'][idx])
    
    def _forward_check(self, outputs, targets, cycle_train=False):
        def _check(a, b):
            assert isinstance(a, (tuple, list, nn.ModuleList)), "{} should be a tuple or list !!!".format(a)
            assert isinstance(b, (tuple, list, nn.ModuleList)), "{} should be  tuple or list !!!".format(b)
            assert len(a) == len(b), "The length is not equal !!!, {} <> {}".format(a, b)

        _check(outputs, targets)
        _check(outputs, self.losses['hm'])
        _check(self.losses['hm'], self.losses['hm_factor'])
        
        if self.with_region_map:
            _check(outputs, self.losses['region'])
            _check(self.losses['region_factor'], self.losses['region'])
        
        if self.with_cycle_detection and cycle_train:
            _check(outputs, self.cd_losses['hm'])
            _check(self.cd_losses['hm'], self.cd_losses['hm_factor'])
            
            if self.with_region_map:
                _check(outputs, self.cd_losses['region'])
                _check(self.cd_losses['region_factor'], self.cd_losses['region'])
     
        

