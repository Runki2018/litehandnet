from .pose_estimation.SRHandNet.SRhandNet import SRHandNet as srhandnet
from .pose_estimation.lite_hrnet import LiteHRNet as litehrnet
from .pose_hg_ms_att import MultiScaleAttentionHourglass as mynet

__all__ = ['srhandnet', 'litehrnet', 'SRHandNet', 'mynet']


def get_model(cfg):
    model_name = cfg.MODEL.name
    assert model_name in __all__, \
        f"model <{model_name}> should be one of {__all__}"

    model = eval(model_name)(cfg)
    return model


