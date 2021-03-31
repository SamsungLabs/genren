import torch
import torch.nn as nn
import torch.nn.functional as F

# From:
# https://github.com/simonepri/PyTorch-BigGraph/blob/b3b1a845e0cc91c750822284a0cadd086dab9413/torchbiggraph/model.py#L601-L610
# Needed because the current version of pytorch cdist fails during the backward pass on GPUs

@torch.jit.script
def batched_cdist_l2sq(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.baddbmm(
            x2_norm.transpose(-2, -1),
            x1,
            x2.transpose(-2, -1),
            alpha=-2
        #).add_(x1_norm).clamp_min_(1e-7) # .sqrt_()
        ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
        return res


#------------------------------------------------#
if __name__ == "__main__":
    device = torch.device(0)
    a = torch.randn(64, 1000, 3).to(device)
    b = torch.randn(64, 700, 3).to(device)
    c1 = torch.cdist(a, b, p = 2)
    c2 = batched_cdist_l2sq(a, b)
    L = (c1 - c2).abs().mean() #.sum()
    print('L', L)
    print('Backward test')
    h = torch.nn.Parameter( b )
    LL = batched_cdist_l2sq(h, a).mean()
    print('LL', LL)
    LL.backward()
    print('\tDone')
#------------------------------------------------#



#
