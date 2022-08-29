
import torch
import torch.nn.functional as F


# ! taken from https://github.com/DengPingFan/Polyp-PVT/blob/main/Train.py
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def cross_entropy(inputs, targets, class_weights=None, pixel_weights=None, ignore_index = -1):
    
    B, C, H, W, = inputs.shape
    B_t, C, H_t, W_t = targets.shape
    assert B==B_t
    assert H_t == H
    assert W == W_t
    
    #import pdb; pdb.set_trace()

    #torch.use_deterministic_algorithms(False)

    loss = F.binary_cross_entropy_with_logits(input=inputs, target= targets, weight=class_weights,
    reduction='mean' if pixel_weights is None else 'none')
    #torch.use_deterministic_algorithms(True)

    if pixel_weights is not None:
        if torch.any(torch.isnan(pixel_weights)):
            print("WARN cross_entropy2d pixel_weights contains NaN. Skip weighting.")
        else:
            loss = pixel_weights.detach() * loss
        loss = torch.sum(loss) / torch.sum(pixel_weights>0)
    return loss


def mse_with_logits(inputs, targets):
    loss = F.mse_loss(inputs.sigmoid(), targets)
    return loss