import torch, torch.nn as nn, torch.nn.functional as F, logging
from networks.imnet_resnets import new_resnet18 

def resnet18_backbone(intermed_dim):
    return nn.Sequential(
                new_resnet18(intermed_dim),
                nn.BatchNorm1d(intermed_dim),
                nn.ReLU() )

class ResShortcutNetwork(nn.Module):
    """
    Mini-resnet. Ends in BN + activation non-linearity.
    """
    def __init__(self, inchannels, outdim, act='lrelu'):
        super(ResShortcutNetwork, self).__init__()
        D = [32, 64, 96, 128, 128]
        end_dim = 4 # H / W at end (num pixels)
        unfolded_size = D[-1] * (end_dim**2)
        logging.info('\tInitialized ResShortcutNetwork')
        logging.info('\t\tLayers: %s\n\t\tUnfolded = Final layer: %d -> %d', 
            str([inchannels] + D), unfolded_size, outdim)
        self.init_block = nn.Sequential( 
                                conv3x3(inchannels, D[0], s=1), # 64 -> 64 [32] <1>
                                #nn.BatchNorm2d(D[0]),
                                get_act(act) )
                                #nn.ReLU() )
        self.resblocks = nn.Sequential(
                                RbDown(D[0], D[1], act), # 64 -> 32 [64]  <3> 
                                RbDown(D[1], D[2], act), # 32 -> 16 [96]  <5> 
                                RbDown(D[2], D[3], act), # 16 -> 8  [128] <7>
                                RbDown(D[3], D[4], act), # 8 -> 4   [128] <9>
                                Unfolder() ) 
        self.final_dense = nn.Sequential(
                                nn.Linear(unfolded_size, outdim),   # <10>
                                nn.BatchNorm1d(outdim),
                                get_act(act) )
                                #nn.ReLU() )

    def forward(self, I):
        return self.final_dense(
                    self.resblocks(
                        self.init_block(I)
                    )
                )

class RbDown(nn.Module):
    def __init__(self, in_channels, out_channels, act):
        super(RbDown, self).__init__()
        self.main = nn.Sequential(
                            conv3x3(in_channels, out_channels, s=2),
                            nn.BatchNorm2d(out_channels),
                            get_act(act),
                            #nn.ReLU(),
                            conv3x3(out_channels, out_channels, s=1),
                            nn.BatchNorm2d(out_channels) )
        self.res_proj = nn.Sequential(
                            conv1x1(in_channels, out_channels, s=2),
                            nn.BatchNorm2d(out_channels) )
        self.a = get_act(act) #,nn.ReLU()

    def forward(self, x):
        return self.a( self.main(x) + self.res_proj(x) )

# No bias since BatchNorm has affine params
def conv1x1(inc, outc, s=1): 
    return nn.Conv2d(inc, outc, kernel_size=1, stride=s, padding=0, bias=False)
def conv3x3(inc, outc, s=1): 
    return nn.Conv2d(inc, outc, kernel_size=3, stride=s, padding=1, bias=False)    

#-----------------------------------------------------------------------------------------#

class Intervalizer(nn.Module):
    def __init__(self, minv, maxv):
        super(Intervalizer, self).__init__()
        delta = maxv - minv
        self.register_buffer('delta', delta)
        self.register_buffer('minv', minv)
        self.register_buffer('maxv', maxv)
        assert all(self.maxv >= self.minv)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(x) # B x D, in [-1,1]
        return (self.delta*(x + 1.0)/2.0) + self.minv

def lin_to_interval(insize, outsize, minv, maxv):
    return nn.Sequential(
                nn.Linear(insize, outsize),
                Intervalizer(minv, maxv) )

def LBA_stack_to_reshape(layer_sizes, 
                         final_shape, 
                         norm_type='batch', 
                         act_type='elu', 
                         init_lin_scale=None, 
                         end_with_lin=True,
                         append_BN=False,
                         append_tanh=False,
                         append_sigmoid=False):
    if append_tanh or append_sigmoid:
        network = nn.Sequential(
                    LBA_stack(layer_sizes, norm_type, act_type, end_with_lin),
                    nn.BatchNorm1d(layer_sizes[-1]) if append_BN else Identity(),
                    Reshaper( final_shape ),
                    nn.Tanh() if append_tanh else nn.Sigmoid() )
    else:
        network = nn.Sequential(
                    LBA_stack(layer_sizes, norm_type, act_type, end_with_lin),
                    nn.BatchNorm1d(layer_sizes[-1]) if append_BN else Identity(),
                    Reshaper( final_shape ) )
    if (not init_lin_scale is None): # or ( abs(init_lin_scale - 1) < 1e-7 ):
        # If the scale is 1, do nothing
        if ( abs(init_lin_scale - 1) < 1e-8 ): return network
        # Otherwise downscale all the weights
        def scale_lin_weights(m):
            namae = m.__class__.__name__
            if namae.find('Linear') != -1:
                m.weight.data *= init_lin_scale
                m.bias.data   *= init_lin_scale
        network.apply(scale_lin_weights)
    return network

def LBA_stack(sizes, norm_type='batch', act_type='elu', end_with_lin=True):
    n = len(sizes)-1
    layers = [ 
               LinBnAct(sizes[i], sizes[i+1], norm_type, act_type)
               for i in range(n-1) 
             ] + [ 
               (  nn.Linear(sizes[n-1], sizes[n])
                  if not (norm_type in ['sn', 'spectral']) else
                  nn.utils.spectral_norm( nn.Linear(sizes[n-1], sizes[n]) )
               )
               if end_with_lin else
               LinBnAct(sizes[n-1], sizes[n], norm_type, act_type) 
             ]
    return nn.Sequential(*layers)

class Reshaper(nn.Module):
    def __init__(self, outshape):
        super(Reshaper, self).__init__()
        self.s = outshape

    def forward(self, x):
        B = x.shape[0]
        return x.view(B, *self.s)

class Unfolder(nn.Module):
    def __init__(self):
        super(Unfolder, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Symmetrizer(nn.Module):
    """
    Computes the mapping X -> X + X^T, to symmetrize a matrix X.
    Input is expected to be B x D x D.
    """
    def __init__(self):
        super(Symmetrizer, self).__init__()

    def forward(self, x):
        return x + x.transpose(1,2)

class Identity(nn.Module):
    """ Performs the identity operation on the input: f(x) = x """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class LinBnAct(nn.Module):
    """
    Simple block applying a linear (actually affine) transform, batchnorm, and
        finally a non-linear activation
    """
    def __init__(self, inlen, outlen, norm_type='batch', act_type=None):
        super(LinBnAct, self).__init__()

        norm_type = norm_type.strip().lower()
        assert norm_type in [ 'spectral', 'batch', 'bn', 'sn', 
                              'layer', 'ln',
                              'batch+spectral', 'bn+sn', 'sn+bn' ]
        if norm_type == 'batch':          norm_type = 'bn'
        if norm_type == 'spectral':       norm_type = 'sn'
        if norm_type == 'layer':          norm_type = 'ln'
        if norm_type == 'batch+spectral': norm_type = 'bn+sn'
        if norm_type == 'sn+bn':          norm_type = 'bn+sn'

        # Default nonlinear activation
        if act_type is None: act_type = 'elu'

        self.din  = inlen
        self.dout = outlen

        if norm_type in ['bn', 'ln']:
            self.f = nn.Sequential(
                        nn.Linear(inlen, outlen),
                        get_norm(norm_type, outlen),
                        get_act(act_type) )
        elif norm_type == 'sn':
            self.f = nn.Sequential(
                        nn.utils.spectral_norm( nn.Linear(inlen, outlen) ),
                        get_act(act_type) )
        elif norm_type == 'bn+sn':
            # NOTE: this is how SAGAN structures its layers:
            # specnorm(conv) --> batchnorm --> non-linearity
            self.f = nn.Sequential(
                        nn.utils.spectral_norm( nn.Linear(inlen, outlen) ),
                        get_norm('bn', outlen),
                        get_act(act_type) )

    def forward(self, x):
        return self.f(x)

class NonVae(nn.Module):
    def __init__(self, network):
        super(NonVae, self).__init__()
        self.f = network

    def forward(self, x):
        return self.f(x), None, None # Fake z, mu, logvar

#----------------------------------------------------------------------------------#

def get_act(name, p=None):
    if name is None or name == 'none': name = 'identity'
    name = name.strip().lower()
    assert name in ['relu', 'sigmoid', 'tanh', 'elu', 'selu', 'lrelu', 'none', 'identity']
    if   name == 'relu':
        return nn.ReLU()
    elif name == 'lrelu' or name == 'leakyrelu':
        if p is None: p = 0.2
        return nn.LeakyReLU(negative_slope=p)
    elif name == 'elu':
        return nn.ELU()
    elif name == 'selu':
        return nn.SELU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'identity':
        return Identity()

def get_norm(name, size=None):
    name = name.strip().lower()
    assert name in ['batch', 'bn', 'layer', 'ln']
    if name == 'batch' or name == 'bn':
        return nn.BatchNorm1d(size)
    if name == 'layer' or name =='ln':
        return nn.LayerNorm(size)






#
