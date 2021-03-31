import torch.nn as nn, torch
import torch.nn.functional as F
from torch import autograd
import logging

from torch_mimicry.nets.wgan_gp import wgan_gp_base
from torch_mimicry.nets.wgan_gp.wgan_gp_resblocks import DBlockOptimized, DBlock, GBlock

### Based on pytorch Mimicry implementation ###

class MicroWGANGPDiscriminator64(wgan_gp_base.WGANGPBaseDiscriminator):
    r"""
    ResNet backbone discriminator for WGAN-GP.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
        gp_scale (float): Lamda parameter for gradient penalty.        
    """
    def __init__(self, inchannels, ndf=1024, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        logging.info('Initializing Micro-WGANGP network (patch critic design)')
        lrel = 0.2
        #dropout = 0.1

        # Build layers
        self.block1 = nn.Sequential(*[
                            nn.utils.spectral_norm( nn.Conv2d(inchannels, 32, kernel_size=3, stride=1, padding=1) ),
                            nn.LeakyReLU(lrel, inplace=False),
                            nn.utils.spectral_norm( nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0) ),
                            nn.LeakyReLU(lrel, inplace=False)
                        ])
        self.block2 = nn.Sequential(*[
                            nn.utils.spectral_norm( nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) ),
                            nn.LeakyReLU(lrel, inplace=False),
                            nn.utils.spectral_norm( nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0) ),
                            nn.LeakyReLU(lrel, inplace=False)
                        ])
        self.block3 = nn.Sequential(*[
                            nn.utils.spectral_norm( nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1) ),
                            nn.LeakyReLU(lrel, inplace=False),
                            nn.utils.spectral_norm( nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0) ),
                            nn.LeakyReLU(lrel, inplace=False)
                        ])

        #self.block1 = DBlockOptimized(inchannels, self.ndf >> 4)
        #self.block2 = DBlock(self.ndf >> 4, self.ndf >> 3, downsample=True)
        #self.block3 = DBlock(self.ndf >> 3, self.ndf >> 2, downsample=True)
        #self.block4 = DBlock(self.ndf >> 2, self.ndf >> 1, downsample=True)
        #self.block5 = DBlock(self.ndf >> 1, self.ndf, downsample=True)
        #self.l6 = nn.Linear(self.ndf, 1)
        self.final_conv = nn.utils.spectral_norm( nn.Conv2d(16, 1, kernel_size=1) ) #, stride=s, padding=p)
        #self.activation = nn.ReLU()

        # Initialise the weights
        #nn.init.xavier_uniform_(self.l6.weight.data, 1.0)

    def forward(self, x, intermeds_only=False):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.
        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).
        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        if intermeds_only:
            h1 = self.block1(x)
            h2 = self.block2(h1)
            h3 = self.block3(h2)
            #h4 = self.block4(h3)
            #h5 = self.block5(h4)
            #return (h1, h2, h3, h4, h5)
            #----#
            # Normalize like E-LPIPS
            h1 = h1 / h1.pow(2).sum(dim=1,keepdim=True).clamp(min=1e-5).sqrt()
            h2 = h2 / h2.pow(2).sum(dim=1,keepdim=True).clamp(min=1e-5).sqrt()
            h3 = h3 / h3.pow(2).sum(dim=1,keepdim=True).clamp(min=1e-5).sqrt()            
            #----#
            return (h1, h2, h3)
        # Normal forward pass
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        #h = self.block4(h)
        #h = self.block5(h)
        #h = self.activation(h)
        h = self.final_conv(h) # replace linear layer
        # Global average pooling
        #h = torch.mean(h, dim=(2, 3))  # WGAN uses mean pooling over space [now 16D vector]
        #output = self.l6(h)
        # Patch critic, means over space and channels
        return h.mean() #

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
            loss += (p *
                        ( a.view(B,NH,nci,hi,wi) -
                          b.view(B,NH,nci,hi,wi)
                        ).abs().mean(dim=-1).mean(dim=-1).mean(dim=-1)                          
                        # ).pow(2).mean(dim=-1).mean(dim=-1).mean(dim=-1)
                    ).sum(dim=1).mean(dim=0)
        return loss


class WGANGPDiscriminator64(wgan_gp_base.WGANGPBaseDiscriminator):
    r"""
    ResNet backbone discriminator for WGAN-GP.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
        gp_scale (float): Lamda parameter for gradient penalty.        
    """
    def __init__(self, inchannels, ndf=1024, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        logging.info('Initializing FULL SIZE WGAN-GP network')

        # Build layers
        self.block1 = DBlockOptimized(inchannels, self.ndf >> 4)
        self.block2 = DBlock(self.ndf >> 4, self.ndf >> 3, downsample=True)
        self.block3 = DBlock(self.ndf >> 3, self.ndf >> 2, downsample=True)
        self.block4 = DBlock(self.ndf >> 2, self.ndf >> 1, downsample=True)
        self.block5 = DBlock(self.ndf >> 1, self.ndf, downsample=True)
        self.l6 = nn.Linear(self.ndf, 1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l6.weight.data, 1.0)

    def forward(self, x, intermeds_only=False):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.
        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).
        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        if intermeds_only:
            h1 = self.block1(x)
            h2 = self.block2(h1)
            h3 = self.block3(h2)
            h4 = self.block4(h3)
            h5 = self.block5(h4)
            #----#
            # Normalize like E-LPIPS
            h1 = h1 / h1.pow(2).sum(dim=1,keepdim=True).clamp(min=1e-5).sqrt()
            h2 = h2 / h2.pow(2).sum(dim=1,keepdim=True).clamp(min=1e-5).sqrt()
            h3 = h3 / h3.pow(2).sum(dim=1,keepdim=True).clamp(min=1e-5).sqrt()            
            h4 = h4 / h4.pow(2).sum(dim=1,keepdim=True).clamp(min=1e-5).sqrt()            
            h5 = h5 / h5.pow(2).sum(dim=1,keepdim=True).clamp(min=1e-5).sqrt()            
            #----#
            return (h1, h2, h3, h4, h5)
        # Normal forward pass
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global average pooling
        h = torch.mean(h, dim=(2, 3))  # WGAN uses mean pooling
        output = self.l6(h)
        return output

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
            loss += (p *
                        ( a.view(B,NH,nci,hi,wi) -
                          b.view(B,NH,nci,hi,wi)
                        ).abs().mean(dim=-1).mean(dim=-1).mean(dim=-1)                          
                        # ).pow(2).mean(dim=-1).mean(dim=-1).mean(dim=-1)
                    ).sum(dim=1).mean(dim=0)
        return loss

def wasserstein_loss_dis(output_real, output_fake, drift_mag_weight):
    r"""
    Computes the wasserstein loss for the discriminator.
    Args:
        output_real (Tensor): Discriminator output logits for real images.
        output_fake (Tensor): Discriminator output logits for fake images.
    Returns:
        Tensor: A scalar tensor loss output.        
    """
    # Estimate Wasserstein distance
    # Push reals high, fakes low
    loss = -1.0 * output_real.mean() + output_fake.mean()
    # Drift penalty, as in StyleGAN
    drift_loss = output_real.pow(2).mean() * drift_mag_weight
    return loss + drift_loss

def wasserstein_loss_gen(output_fake):
    r"""
    Computes the wasserstein loss for generator.
    Args:
        output_fake (Tensor): Discriminator output logits for fake images.
    Returns:
        Tensor: A scalar tensor loss output.
    """
    # Push fakes high
    loss = -output_fake.mean()
    return loss

def compute_gradient_penalty_loss_texture(model,
                                          reals,
                                          fakes,
                                          gp_scale # 10
                                          ):
        r"""
        Computes gradient penalty loss, as based on:
        https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
        
        Args:
            reals (Tensor): A batch of real vectors of shape (N, M, d).
            fakes (Tensor): A batch of fake vectors of shape (N, M, d).
            gp_scale (float): Gradient penalty lamda parameter.
        Returns:
            Tensor: Scalar gradient penalty loss.
        """
        # Obtain parameters
        N, M, D = reals.shape
        device = reals.device

        # Randomly sample some alpha between 0 and 1 for interpolation
        # where alpha is of the same shape for elementwise multiplication.
        alpha = torch.rand(N, 1, 1)
        alpha = alpha.expand(-1, M, D).contiguous()
        alpha = alpha.view(N, M, D)
        alpha = alpha.to(device)

        # Obtain interpolates on line between real/fake images.
        interpolates = alpha * reals.detach() \
                        + ((1 - alpha) * fakes.detach())
        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)

        # Get gradients of interpolates
        disc_interpolates = model.forward(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size()).to(device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.reshape(gradients.size(0), -1)

        # Compute GP loss
        gradient_penalty = (
            (gradients.norm(2, dim=1) - 1)**2).mean() * gp_scale

        return gradient_penalty

def compute_gradient_penalty_loss_vecs(model,
                                      	real_vecs,
                                      	fake_vecs,
                                      	gp_scale # 10
                                      	):
        r"""
        Computes gradient penalty loss, as based on:
        https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
        
        Args:
            real_vecs (Tensor): A batch of real vectors of shape (N, d).
            fake_vecs (Tensor): A batch of fake vectors of shape (N, d).
            gp_scale (float): Gradient penalty lamda parameter.
        Returns:
            Tensor: Scalar gradient penalty loss.
        """
        # Obtain parameters
        N, D = real_vecs.shape
        device = real_vecs.device

        # Randomly sample some alpha between 0 and 1 for interpolation
        # where alpha is of the same shape for elementwise multiplication.
        alpha = torch.rand(N, 1)
        alpha = alpha.expand(N, int(real_vecs.nelement() / N)).contiguous()
        alpha = alpha.view(N, D)
        alpha = alpha.to(device)

        # Obtain interpolates on line between real/fake images.
        interpolates = alpha * real_vecs.detach() \
            			+ ((1 - alpha) * fake_vecs.detach())
        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)

        # Get gradients of interpolates
        disc_interpolates = model.forward(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size()).to(device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute GP loss
        gradient_penalty = (
            (gradients.norm(2, dim=1) - 1)**2).mean() * gp_scale

        return gradient_penalty


def compute_gradient_penalty_loss_imgs(model,
                                      	real_images,
                                      	fake_images,
                                      	gp_scale): # =10.0):
        r"""
        Computes gradient penalty loss, as based on:
        https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
        
        Args:
            real_images (Tensor): A batch of real images of shape (N, 3, H, W).
            fake_images (Tensor): A batch of fake images of shape (N, 3, H, W).
            gp_scale (float): Gradient penalty lamda parameter.
        Returns:
            Tensor: Scalar gradient penalty loss.
        """
        # Obtain parameters
        N, C, H, W = real_images.shape
        device = real_images.device

        # Randomly sample some alpha between 0 and 1 for interpolation
        # where alpha is of the same shape for elementwise multiplication.
        alpha = torch.rand(N, 1)
        alpha = alpha.expand(N, int(real_images.nelement() / N)).contiguous()
        alpha = alpha.view(N, C, H, W)
        alpha = alpha.to(device)

        # Obtain interpolates on line between real/fake images.
        interpolates = alpha * real_images.detach() \
            + ((1 - alpha) * fake_images.detach())
        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)

        # Get gradients of interpolates
        disc_interpolates = model.forward(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size()).to(device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute GP loss
        gradient_penalty = (
            (gradients.norm(2, dim=1) - 1)**2).mean() * gp_scale

        return gradient_penalty


#

