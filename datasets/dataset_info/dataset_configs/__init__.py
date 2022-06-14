from .coco import dataset_info as coco_info
from .coco_wholebody import dataset_info as coco_wholebody_info
from .coco_wholebody_hand import dataset_info as coco_wholebody_hand_info
from .freihand2d import dataset_info as freihand2d_info
from .halpe import dataset_info as halpe_info
from .mpii import dataset_info as mpii_info
from .zhhand import dataset_info as zhhand_info
from .rhd2d import dataset_info as rhd2d_info
from .panoptic_hand2d import dataset_info as panoptic_info

#  # 封装Python字典，可以'.'访问键
# coco_info = addict.Dict(coco_info)
# coco_wholebody_info = addict.Dict(coco_wholebody_info)
# coco_wholebody_hand_info = addict.Dict(coco_wholebody_hand_info)
# freihand2d_info = addict.Dict(freihand2d_info)
# halpe_info = addict.Dict(halpe_info)
# mpii_info = addict.Dict(mpii_info)
# zhhand_info = addict.Dict(zhhand_info)

__all__ = [
    'coco_info', 'coco_wholebody_info', 'coco_wholebody_hand_info',
    'freihand2d_info', 'halpe_info', 'mpii_info', 'zhhand_info', 'rhd2d_info',
    'panoptic_info'
]