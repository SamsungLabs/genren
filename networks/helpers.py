import torch, torch.nn as nn, torch.nn.functional as F

#################
# Interpolation #
#################

def DoublingUpscaler(mode='bilinear'):
    return Interpolator(scale_factor=2, interp_type=mode)

def HalvingDownscaler(mode='bilinear'):
    return Interpolator(scale_factor=0.5, interp_type=mode)

class Interpolator(nn.Module):
    def __init__(self, scale_factor, interp_type='bilinear', align_corners=True):
        super(Interpolator, self).__init__()
        assert interp_type in ['bilinear', 'nearest', 'bicubic'], 'Unknown ' + str(interp_type)

        self.mode = interp_type
        self.s = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.s, mode=self.mode,
                             align_corners=self.align_corners)

#############
# Image Ops #
#############

class GlobalMean(nn.Module):
    def __init__(self):
        super(GlobalMean, self).__init__()

    def forward(self, x):
        # x is B x C x H x W -> output is B x C
        return x.mean(dim=-1).mean(dim=-1)








#
