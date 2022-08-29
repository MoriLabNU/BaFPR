from .pvt import PolypPVT
from .BaFPR import BaFPR
#from .UACANet import UACANet

model_dict = {
   'PolypPVT': PolypPVT,
   'BaFPR': BaFPR,
   #'UACA_L':UACANet,
}


def _get_model(cfg):
    assert cfg.model.backbone in model_dict.keys()
    return model_dict[cfg.model.backbone]
