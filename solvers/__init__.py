import copy

from .losses import structure_loss, cross_entropy, mse_with_logits
from .optimizers import AdamW
from .schedulers import adjust_lr

optimizer_dict ={
    'AdamW': AdamW,
}


def _get_optim(cfg):
    assert cfg.optimizer.type in optimizer_dict.keys()
    
    return optimizer_dict[cfg.optimizer.type]
