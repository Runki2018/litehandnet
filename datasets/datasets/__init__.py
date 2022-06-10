from .body.topdown_coco_dataset import TopDownCocoDataset  as coco
from .body.topdown_mpii_dataset import TopDownMpiiDataset as mpii
from .hand.freihand_dataset import FreiHandDataset as freihand
from .hand.coco_wholebody_hand_dataset import HandCocoWholeBodyDataset as coco_wholebody_hand
from .hand.zhhand_dataset import ZHHandDataset as zhhand


__all__ = [
    'coco', 'mpii', 'freihand', 'coco_wholebody_hand', 'zhhand'
]