"""
Modified from:
https://github.com/godisboy/SN-GAN
"""

import torch.nn as nn, torch
import torch.nn.functional as F

def SNConv2d(in_channels, hidden_channels, kernel_size=3, padding=1, stride=1):
    return nn.utils.spectral_norm( 
        nn.Conv2d(in_channels, hidden_channels, 
                kernel_size=kernel_size, stride=stride, padding=padding) 
    )

def SNLinear(insize, outsize):
    return nn.utils.spectral_norm( 
        nn.Linear(insize, outsize)
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, use_BN = False, downsample=False):
        super(ResBlock, self).__init__()
        #self.conv1 = SNConv2d(n_dim, n_out, kernel_size=3, stride=2)
        hidden_channels = in_channels
        self.downsample = downsample

        self.resblock = self.make_res_block(in_channels, out_channels, hidden_channels, use_BN, downsample)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)
    def make_res_block(self, in_channels, out_channels, hidden_channels, use_BN, downsample):
        model = []
        if use_BN:
            model += [nn.BatchNorm2d(in_channels)]

        model += [nn.ReLU()]
        model += [SNConv2d(in_channels, hidden_channels, kernel_size=3, padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(hidden_channels, out_channels, kernel_size=3, padding=1)]
        if downsample:
            model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)
    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        if self.downsample:
            model += [nn.AvgPool2d(2)]
            return nn.Sequential(*model)
        else:
            return nn.Sequential(*model)

    def forward(self, input):
        return self.resblock(input) + self.residual_connect(input)

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OptimizedBlock, self).__init__()
        self.res_block = self.make_res_block(in_channels, out_channels)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)
    def make_res_block(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(out_channels, out_channels, kernel_size=3, padding=1)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)
    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)
    def forward(self, input):
        return self.res_block(input) + self.residual_connect(input)

class SNResDiscriminator(nn.Module):
    def __init__(self, ndf=64, ndlayers=3, n_input_channels=4):
        # ndf = first layer channels
        super(SNResDiscriminator, self).__init__()
        endmult = 2**ndlayers # 64 * 8 = 512
        self.end_dim = ndf * endmult
        #self.res_d = self.make_model(ndf, ndlayers)
        self.b1 = OptimizedBlock(n_input_channels, ndf) # 32, 64 [No relu at start or end]
        # i = 0
        tndf = ndf # 16, 128 [Relu at start, not end]
        self.b2 = ResBlock(tndf,   tndf*2, downsample=True)
        # i = 1
        tndf *= 2 # 8, 256 [Relu at start, not end]
        self.b3 = ResBlock(tndf,   tndf*2, downsample=True)
        # i = 2
        tndf *= 2 # 4, 512 [Relu at start, not end]
        self.b4 = ResBlock(tndf,   tndf*2, downsample=True)
        # i = 3, 4x4, 512 -> 4x4, 512 [Relu at start AND end]
        self.b5 = nn.Sequential( 
                        ResBlock(tndf*2, tndf*2, downsample=False),
                        nn.ReLU() )
        #self.fc = nn.Sequential(SNLinear(ndf*16, 1), nn.Sigmoid())
        self.fc = SNLinear(self.end_dim, 1)

    def res_d(self, x):
        return self.b5( self.b4( self.b3( self.b2( self.b1(x) ) ) ) ) 

    def forward(self, input, intermeds_only=False):
        if intermeds_only:
            out1 = self.b1(input)
            out2 = self.b2(out1)
            out3 = self.b3(out2)
            out4 = self.b4(out3)
            return out1, out2, out3, out4
        out = self.res_d(input) # Output is relu-ed
        out = F.avg_pool2d(out, out.size(3), stride=1)
        out = out.view(-1, self.end_dim)
        return self.fc(out)




#
