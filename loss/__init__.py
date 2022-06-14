from loss.heatmapLoss import (RegionLoss, MaskLoss,  JointsDistanceLoss,
                              KLFocalLoss, FocalLoss, DistanceLoss)
from loss.centernet_simdr_loss import KLDiscretLoss, SimDRLoss


from .loss import SRHandNetLoss as srhandnetloss
from .loss import TopdownHeatmapLoss as topdownheatmaploss



__all__ = [
    'RegionLoss', 'MaskLoss', 'JointsDistanceLoss', 'DistanceLoss'
     'KLFocalLoss', 'KLDiscretLoss', 'FocalLoss', 'srhandnetloss',
     'topdownheatmaploss', 'SimDRLoss'
]


def get_loss(cfg):
    return eval(cfg.LOSS.type.lower())(cfg)