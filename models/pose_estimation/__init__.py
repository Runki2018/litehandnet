from .pose_hg_ms_att import MultiScaleAttentionHourglass as hg_ms_att
from .SRhandNet import get_SRhandNet as srhandnet
from .lite_hrnet import LiteHRNet as litehrnet

__all__ = ['hg_ms_att', 'srhandnet', 'litehrnet']


def get_model(cfg):
    model_name = cfg['model']
    assert model_name in __all__, "model <{}> should be one of {} !!!".format(model_name, __all__)     
    model = eval(model_name)(cfg)
    return model




     