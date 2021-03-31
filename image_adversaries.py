import torch, torch.nn as nn, torch.nn.functional as F, logging
from torch import autograd
from torch.autograd import Variable
from networks.helpers import HalvingDownscaler, GlobalMean

class RotationPredictorRN20(nn.Module):
    """
    Predicts a rotation matrix from an input texture image
    """
    def __init__(self):
        super(RotationPredictorRN20, self).__init__()
        from networks.cifar_resnets import resnet20
        self.predictor = resnet20(output_dim = 4)
        from graphicsutils import QuatRotationDecoder, MinAngleComposedRotationLoss
        self.quat_to_mat = QuatRotationDecoder()
        self.rotmat_loss = MinAngleComposedRotationLoss()

    def forward(self, TI):
        """ Map a texture image TI to a rotation matrix """
        return self.quat_to_mat(self.predictor(TI))

    def loss(self, TI, R_true):
        return self.rotmat_loss(self(TI), R_true)


class ImageAdversarySimpleHingeGan(nn.Module):
    def __init__(self):
        super(ImageAdversarySimpleHingeGan, self).__init__()
        logging.info('Using Hinge-SNGAN for an image critic')
        self.f = MiniSpecGanCritic()

    def forward(self, for_gen, I_fake, I_real=None, 
                compute_hyp_loss=False, I1=None, I2=None, p=None):
        # Case 1: generator loss
        if for_gen:
            return self.compute_loss_for_generator(I_fake)
        # Case 2: perceptual distance
        if compute_hyp_loss: 
            return self.hyp_weighted_perceptual_loss(I1, I2, p)
        # Case 3: critic loss
        return hinge_loss_dis(self.f(I_fake), self.f(I_real))

    def compute_loss_for_generator(self, I):
        return hinge_loss_gen(self.f(I)) 

    def hyp_weighted_perceptual_loss(self, I1, I2, p):
        return self.f.hyp_weighted_perceptual_loss(I1, I2, p)

class ImageAdversarySimpleLSGAN(nn.Module):

    def __init__(self, critic_arch): #, version=2):
        super(ImageAdversarySimpleLSGAN, self).__init__()
        assert critic_arch in ['3scale', 'spec', 'patch']
        self.mse = torch.nn.MSELoss()

        logging.info('Using Simple Image Adversary (SN-LSGAN)')
        if critic_arch == '3scale':
            logging.info('\tUsing 3-scale critic')
            self.f = SnCritic3S()
        elif critic_arch == 'spec':
            logging.info('\tUsing mini-specgan-critic')
            self.f = MiniSpecGanCritic()
        elif critic_arch == 'patch':
            logging.info('\tUsing mini-patch critic')
            from networks.wgan_helpers import MicroWGANGPDiscriminator64 as WGANGP_net # naming is... not so good here
            self.f = WGANGP_net(inchannels=4) 

    def forward(self, for_gen, I_fake, I_real=None, noise_eps=0.01, 
                compute_hyp_loss=False, I1=None, I2=None, p=None):
        # Case 1: generator loss
        if for_gen:
            return self.compute_loss_for_generator(I_fake)
        # Case 2: perceptual distance
        if compute_hyp_loss: 
            return self.hyp_weighted_perceptual_loss(I1, I2, p)
        # Case 3: critic loss
        device = I_fake.device
        #eps = noise_eps
        #I_fake = torch.clamp(I_fake + eps * torch.randn(I_fake.shape).to(device),
        #                     min=-1.0, max=1.0)
        #I_real = torch.clamp(I_real + eps * torch.randn(I_real.shape).to(device),
        #                     min=-1.0, max=1.0)
        #I_fake, I_real = I_fake[:,0:3,:,:], I_real[:,0:3,:,:]
        valid = Variable(torch.Tensor(I_fake.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(torch.Tensor(I_real.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
        real_loss = self.mse( self.f(I_real), valid) # Push reals to one
        fake_loss = self.mse( self.f(I_fake.detach()), fake) # Push fakes to zero
        d_loss = 0.5 * (real_loss + fake_loss)
        return d_loss

    def compute_loss_for_generator(self, I):
        return (self.f(I) - 1.0).pow(2).mean() * 0.5

    def hyp_weighted_perceptual_loss(self, I1, I2, p):
        return self.f.hyp_weighted_perceptual_loss(I1, I2, p)

from networks.wgan_helpers import *

class ImageAdversarySimpleWganGp(nn.Module):
    """
    D_w = D_real - D_fake
    Loss for critic: D_fake - D_real + gradpen // push reals to large pos vals, fakes to small (or neg) vals
    Loss for generator: - D_fake // push fakes score to be large and positive vals
    """
    def __init__(self, lambda_weight, drift_mag_weight, critic_type, inchannels=4):
        super(ImageAdversarySimpleWganGp, self).__init__()
        assert critic_type in ['mini', 'full']

        logging.info('Using Simple Image Adversary (WGAN-GP)-%s' % critic_type)

        if   critic_type == 'mini':
            from networks.wgan_helpers import MicroWGANGPDiscriminator64 as WGANGP_net
        elif critic_type == 'full':
            from networks.wgan_helpers import WGANGPDiscriminator64 as WGANGP_net
        self.f = WGANGP_net(inchannels=inchannels) # 

        self.lambda_weight = lambda_weight
        self.drift_mag_weight = drift_mag_weight
        self.noise_sigma = None # 0.01

    def forward(self, for_gen, I_fake, I_real=None, compute_hyp_loss=False, I1=None, I2=None, p=None):
        #I_fake = I_fake[:,0:3,:,:] if not I_fake is None else None
        if for_gen:
            return self.compute_loss_for_generator(I_fake)
        elif compute_hyp_loss:
            assert len(I1.shape) == 5 and len(I2.shape) == 5
            #I1 = I1[:,:,0:3,:,:]
            #I2 = I2[:,:,0:3,:,:]
            return self.hyp_weighted_perceptual_loss(I1, I2, p)
        else:
            #I_real = I_real[:,0:3,:,:]
            if not self.noise_sigma is None:
                device = I_fake.device
                #I_fake = torch.clamp(I_fake + self.noise_sigma * torch.randn(I_fake.shape).to(device),
                #                     min = -1.0, max = 1.0)
                #I_real = torch.clamp(I_real + self.noise_sigma * torch.randn(I_real.shape).to(device),
                #                     min = -1.0, max = 1.0)
            L1 = wasserstein_loss_dis(self.f(I_real), self.f(I_fake), 
                                      drift_mag_weight = self.drift_mag_weight)
            L2 = compute_gradient_penalty_loss_imgs(self.f, I_real, I_fake, self.lambda_weight)
            return L1 + L2

    def compute_loss_for_generator(self, I):
        return wasserstein_loss_gen(self.f(I))

    def hyp_weighted_perceptual_loss(self, I1, I2, p):
        return self.f.hyp_weighted_perceptual_loss(I1, I2, p)

class ImageAdversarySimpleWganGpOLD(nn.Module):
    """
    D_w = D_real - D_fake
    Loss for critic: D_fake - D_real + gradpen // push reals to large pos vals, fakes to small (or neg) vals
    Loss for generator: - D_fake // push fakes score to be large and positive vals
    """
    def __init__(self, lambda_weight):
        super(ImageAdversarySimpleWganGp, self).__init__()
        self.f = WganGpDiscriminator(3)
        self.w_lambda = lambda_weight

    def forward(self, for_gen, I_fake, I_real=None):
        if for_gen:
            return self.compute_loss_for_generator(I_fake)
        else:
            I_fake = I_fake[:,0:3,:,:]
            I_real = I_real[:,0:3,:,:]
            loss_real = self.f(I_real)
            loss_fake = self.f(I_fake)
            gradpen = calculate_gradient_penalty(self.f, I_real.data, I_fake.data)
            return loss_fake - loss_real + self.w_lambda * gradpen

    def compute_loss_for_generator(self, I):
        I = I[:,0:3,:,:]
        return -1.0 * self.f(I)

    def get_optimizer(self, weight_decay):
        default_lr = 1e-4
        default_b1 = 0.5
        default_b2 = 0.999
        return torch.optim.Adam(self.parameters(), lr=default_lr, 
                betas=(default_b1, default_b2), weight_decay=weight_decay)

###################################################################################################################

class LsganDiscriminator1(nn.Module):
    def __init__(self, img_size, channels=3):
        super(LsganDiscriminator1, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True, dropout=True):
            block = [ nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                      nn.LeakyReLU(0.2, inplace=True)
                    ] + ([ nn.Dropout2d(0.25) ] if dropout else [])
            if bn: block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def SnCritic3S(c=4):
    return ThreeScaleCritic(channels=c, norm_type='sn')

def BnCritic3S(c=4):
    return ThreeScaleCritic(channels=c, norm_type='bn')

class SN_Cnn_Critic(nn.Module):
    """
    Inspired by the "classical" architecture (i.e., non-resnet) of the SN-GAN critic.
    """
    def __init__(self):
        super(SN_Cnn_Critic, self).__init__()
        logging.info('Using SN-CNN critic')

    def forward(self, x, intermeds_only=False):
        mini_img = self.halver(img)
        if intermeds_only:
            out1 = self.c1(img, intermeds_only=True) # On full image
            out2 = self.c2(mini_img, intermeds_only=True) # On downscaled image
            out3 = self.c3(self.halver(mini_img), intermeds_only=True) # On double downscaled image
            return [*out1, *out2, *out3]
        else:
            out1 = self.c1(img) # On full image
            out2 = self.c2(mini_img) # On downscaled image
            out3 = self.c3(self.halver(mini_img)) # On double downscaled image
            return (out1 + out2 + out3) / 3.0
    
    def hyp_weighted_perceptual_loss(self, I1, I2, p):
        """
        Args:
            I1: B x NH x NC x H x W
            I2: same as I1
            p: B x NH
        """
        I1 = I1[:, :, 0:3, :, :]
        I2 = I2[:, :, 0:3, :, :]
        B, NH, NC, H, W = I1.shape
        loss = 0.0
        I1 = self(I1.reshape(B*NH,NC,H,W), intermeds_only=True)
        I2 = self(I2.reshape(B*NH,NC,H,W), intermeds_only=True)
        for a, b in zip(I1, I2):
            BNH, nci, hi, wi = a.shape
            # a,b are B*NH * NC_i * H_i * W_i
            loss += (p *
                        ( a.view(B,NH,nci,hi,wi) -
                          b.view(B,NH,nci,hi,wi)
                        ).pow(2).mean(dim=-1).mean(dim=-1).mean(dim=-1)
                    ).sum(dim=1).mean(dim=0)
        return loss

class MiniSpecGanCritic(nn.Module):
    def __init__(self):
        super(MiniSpecGanCritic, self).__init__()
        from networks.sngan_helpers import SNResDiscriminator
        self.d = SNResDiscriminator()

    def forward(self, x, intermeds_only=False):
        if intermeds_only:
            return self.d(x, intermeds_only)
        return self.d(x)

    def hyp_weighted_perceptual_loss(self, I1, I2, p):
        """
        Args:
            I1: B x NH x NC x H x W
            I2: same as I1
            p: B x NH
        """
        #I1 = I1[:, :, 0:3, :, :]
        #I2 = I2[:, :, 0:3, :, :]
        B, NH, NC, H, W = I1.shape
        loss = 0.0
        I1 = self(I1.reshape(B*NH,NC,H,W), intermeds_only=True)
        I2 = self(I2.reshape(B*NH,NC,H,W), intermeds_only=True)
        for a, b in zip(I1, I2):
            BNH, nci, hi, wi = a.shape
            # a,b are B*NH * NC_i * H_i * W_i
            loss += (p * ( a.view(B, NH, nci, hi, wi) - 
                           b.view(B, NH, nci, hi, wi) 
                         ).pow(2).mean(dim=-1).mean(dim=-1).mean(dim=-1)  
                    ).sum(dim=1).mean(dim=0)
        return loss


class ThreeScaleCritic(nn.Module):
    """
    Modified from/inspired by:
        "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs"
    """
    def __init__(self, channels, norm_type, img_size=None) :
        super(ThreeScaleCritic, self).__init__()

        DROPOUT_PROB = 0.2

        assert norm_type in ['sn', 'in', 'bn']
        if norm_type == 'sn': print("\tUsing SpectralNorm")
        if norm_type == 'bn': print("\tUsing BatchNorm")

        _Cs = [ 64, 128, 256, 512 ]

        def one_scale_critic():
            return OneScaleCritic([channels] + _Cs, norm_type=norm_type, dropout_prob=DROPOUT_PROB)

        #def one_scale_critic():
        #    return nn.Sequential(
        #                *discriminator_block2(channels, _Cs[0], norm=False, dropout=False, s=2), # 3
        #                *discriminator_block2(_Cs[0], _Cs[1], s=1), # 7
        #                *discriminator_block2(_Cs[1], _Cs[2], s=1), # 11
        #                *discriminator_block2(_Cs[2], _Cs[3], s=1), # 15
        #                nn.Conv2d(_Cs[3], 1, kernel_size=1),
        #                GlobalMean()
        #                )

        self.halver = HalvingDownscaler()
        self.c1 = one_scale_critic()
        self.c2 = one_scale_critic()
        self.c3 = one_scale_critic()

    def forward(self, img, intermeds_only=False):
        mini_img = self.halver(img)
        if intermeds_only:
            out1 = self.c1(img, intermeds_only=True) # On full image
            out2 = self.c2(mini_img, intermeds_only=True) # On downscaled image
            out3 = self.c3(self.halver(mini_img), intermeds_only=True) # On double downscaled image
            return [*out1, *out2, *out3]
        else:
            out1 = self.c1(img) # On full image
            out2 = self.c2(mini_img) # On downscaled image
            out3 = self.c3(self.halver(mini_img)) # On double downscaled image
            return (out1 + out2 + out3) / 3.0

    def perceptual_loss(self, I1, I2):
        loss = 0.0
        I1 = self(I1, intermeds_only=True)
        I2 = self(I2, intermeds_only=True)
        for a,b in zip(I1, I2):
            loss += (a - b).pow(2).mean()
        return loss

    def hyp_weighted_perceptual_loss(self, I1, I2, p):
        """
        Args:
            I1: B x NH x NC x H x W
            I2: same as I1
            p: B x NH
        """
        #I1 = I1[:, :, 0:3, :, :]
        #I2 = I2[:, :, 0:3, :, :]
        B, NH, NC, H, W = I1.shape
        loss = 0.0
        I1 = self(I1.reshape(B*NH,NC,H,W), intermeds_only=True)
        I2 = self(I2.reshape(B*NH,NC,H,W), intermeds_only=True)
        for a, b in zip(I1, I2):
            BNH, nci, hi, wi = a.shape
            # a,b are B*NH * NC_i * H_i * W_i
            loss += (p * ( a.view(B,NH,nci,hi,wi) - 
                           b.view(B,NH,nci,hi,wi) 
                         ).pow(2).mean(dim=-1).mean(dim=-1).mean(dim=-1)  
                    ).sum(dim=1).mean(dim=0)
        return loss


class OneScaleCritic(nn.Module):
    def __init__(self, Cs, norm_type, dropout_prob):
        super(OneScaleCritic, self).__init__()
        channels = Cs[0]
        _Cs = Cs[1:]
        self.block_1 = discriminator_block_osc(channels, _Cs[0], norm_type=norm_type, 
                            dropout_prob=dropout_prob, norm=False, dropout=True, s=2)
        self.block_2 = discriminator_block_osc(_Cs[0],   _Cs[1], norm_type=norm_type, 
                            dropout_prob=dropout_prob, norm=True,  dropout=True, s=1)
        self.block_3 = discriminator_block_osc(_Cs[1],   _Cs[2], norm_type=norm_type, 
                            dropout_prob=dropout_prob, norm=True,  dropout=True, s=1)
        self.block_4 = discriminator_block_osc(_Cs[2],   _Cs[3], norm_type=norm_type, 
                            dropout_prob=dropout_prob, norm=True,  dropout=True, s=1)
        self.project_avg = nn.Sequential(nn.Conv2d(_Cs[3], 1, kernel_size=1), GlobalMean())

    def forward(self, I, intermeds_only=False):
        if intermeds_only: # Use for perceptual loss
            I1 = self.block_1(I)
            I2 = self.block_2(I1)
            I3 = self.block_3(I2)
            I4 = self.block_4(I3)
            return I1, I2, I3, I4
        else:
            return self.project_avg( self.block_4( self.block_3( self.block_2( self.block_1(I) ) ) ) )


def discriminator_block_osc(in_filters, out_filters, norm_type, dropout_prob, norm=True, 
                            dropout=True, s=2, make_seq=True, lrel=0.2):
    assert norm_type in ['sn', 'bn', 'in']
    p = 1 # if s == 2 else 2
    if norm_type == 'sn':
        block = (  [ nn.utils.spectral_norm( 
                        nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=s, padding=p) 
                     ) ] + 
                   #([ _normer(out_filters) ] if norm else []) +
                   [ nn.LeakyReLU(lrel, inplace=False) ] + 
                   ([ nn.Dropout2d(dropout_prob) ] if dropout else [])
                )
    else:
        if not (norm_type=='bn'): assert norm_type == 'in'
        _normer = nn.BatchNorm2d if norm_type=='bn' else nn.InstanceNorm2d
        block = (  [ nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=s, padding=p) ] + 
                   ([ _normer(out_filters) ] if norm else []) +
                   [ nn.LeakyReLU(lrel, inplace=False) ] + 
                   ([ nn.Dropout2d(dropout_prob) ] if dropout else [])
                )
    #if bn: block.append(nn.BatchNorm2d(out_filters, 0.8))
    return nn.Sequential(*block) if make_seq else block


##################################################################################################################

def hinge_loss_dis(output_fake, output_real):
    r"""
    Hinge loss for discriminator.
    Args:
        output_fake (Tensor): Discriminator output logits for fake images.
        output_real (Tensor): Discriminator output logits for real images.
    Returns:
        Tensor: A scalar tensor loss output.        
    """
    loss = F.relu(1.0 - output_real).mean() + \
           F.relu(1.0 + output_fake).mean()
    return loss


def hinge_loss_gen(output_fake):
    r"""
    Hinge loss for generator.
    Args:
        output_fake (Tensor): Discriminator output logits for fake images.
    Returns:
        Tensor: A scalar tensor loss output.      
    """
    loss = -output_fake.mean()
    return loss

########
if __name__ == "__main__":
    critic = ImageAdversarySimpleLSGAN()


#
