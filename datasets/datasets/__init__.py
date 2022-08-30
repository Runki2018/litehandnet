from .body.topdown_coco_dataset import TopDownCocoDataset  as coco
from .body.topdown_mpii_dataset import TopDownMpiiDataset as mpii
from .body.topdown_mpii_action_dataset import TopDownMpiiActionDataset as mpii_action

from .hand.freihand_dataset import FreiHandDataset as freihand
from .hand.coco_wholebody_hand_dataset import HandCocoWholeBodyDataset as coco_wholebody_hand
from .hand.zhhand_dataset import ZHHandDataset as zhhand
from .hand.panoptic_hand2d_dataset import PanopticDataset as panoptic
from .hand.rhd_dataset import RHD2dDataset as rhd
from .hand.onehand10k_dataset import OneHand10KDataset as onehand10k



__all__ = [
    'coco', 'mpii', 'mpii_action','freihand', 'coco_wholebody_hand', 'zhhand', 'panoptic',
    'rhd', 'onehand10k'
]