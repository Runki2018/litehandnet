from .weight_init import normal_init, kaiming_init, constant_init

from .pose_estimation.SRHandNet.SRhandNet import SRHandNet as srhandnet
from .pose_estimation.lite_hrnet import LiteHRNet as litehrnet
from .pose_estimation.SimpleBaseline.resnet import PoseResNet as resnet
from .pose_estimation.SimpleBaseline.mobilenetv2 import PoseMobileNetV2 as mobilenetv2
from .pose_estimation.hourglassnet import HourglassNet as hourglass
from .pose_estimation.AttentionHandNet import light_Model as atthandnet

from .hourglass_ablation import  hourglass_ablation
from .pose_estimation.liteHandNet.liteHandNet import LiteHandNet as litehandnet
from .pose_hg_ms_att import MultiScaleAttentionHourglass as mynet


__all__ = ['normal_init', 'kaiming_init', 'constant_init',
           'srhandnet', 'litehrnet',  'mynet', 'litehandnet', 'SRHandNet',
           'resnet', 'mobilenetv2', 'hourglass', 'hourglass_ablation', 'atthandnet']


def get_model(cfg):
    model_name = cfg.MODEL.name
    assert model_name in __all__, \
        f"model <{model_name}> should be one of {__all__}"

    model = eval(model_name)(cfg)
    return model


