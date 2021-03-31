import torch, torch.optim as optim, torch.nn as nn, torch.nn.functional as F
import torch_mimicry as mmc, argparse, os, sys, imgutils, re
from torch_mimicry.nets import sngan
from options import get_options
from torch import autograd

from torch_mimicry.nets.wgan_gp import wgan_gp_base
from torch_mimicry.nets.wgan_gp.wgan_gp_resblocks import DBlockOptimized, DBlock, GBlock

class WGANGPGenerator32(wgan_gp_base.WGANGPBaseGenerator):
    r"""
    ResNet backbone generator for WGAN-GP.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, nz=128, ngf=1024, bottom_width=4, outchannels=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        print('Building WGANGP generator with cout = %d' % outchannels)

        # Build the layers
        self.l1     = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf >> 1, upsample=True)
        self.block3 = GBlock(self.ngf >> 1, self.ngf >> 2, upsample=True)
        self.block4 = GBlock(self.ngf >> 2, self.ngf >> 3, upsample=True)
        #self.block5 = GBlock(self.ngf >> 3, self.ngf >> 4, upsample=True)
        self.b6     = nn.BatchNorm2d(self.ngf >> 3)
        self.c6     = nn.Conv2d(self.ngf >> 3, outchannels, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.
        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).
        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        #print('rr', x.shape)
        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        #h = self.block5(h)
        h = self.b6(h)
        h = self.activation(h)
        h = torch.tanh(self.c6(h))

        #print('hh', h.shape)

        return h

class WGANGPGenerator64(wgan_gp_base.WGANGPBaseGenerator):
    r"""
    ResNet backbone generator for WGAN-GP.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, nz=128, ngf=1024, bottom_width=4, outchannels=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        print('Building WGANGP generator with cout = %d' % outchannels)

        # Build the layers
        self.l1     = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf >> 1, upsample=True)
        self.block3 = GBlock(self.ngf >> 1, self.ngf >> 2, upsample=True)
        self.block4 = GBlock(self.ngf >> 2, self.ngf >> 3, upsample=True)
        self.block5 = GBlock(self.ngf >> 3, self.ngf >> 4, upsample=True)
        self.b6     = nn.BatchNorm2d(self.ngf >> 4)
        self.c6     = nn.Conv2d(self.ngf >> 4, outchannels, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.
        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).
        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        #print('rr', x.shape)
        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.b6(h)
        h = self.activation(h)
        h = torch.tanh(self.c6(h))

        #print('hh', h.shape)

        return h

class WGANGPDiscriminator64(wgan_gp_base.WGANGPBaseDiscriminator):
    r"""
    ResNet backbone discriminator for WGAN-GP.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
        gp_scale (float): Lamda parameter for gradient penalty.        
    """
    def __init__(self, ndf=1024, inchannels=4, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        print('Building WGANGP critic with Cin = %d' % inchannels)

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

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.
        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).
        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        #print(x.shape,'x')
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

    def compute_gradient_penalty_loss(self,
                                      real_images,
                                      fake_images,
                                      gp_scale=10.0):
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
        N, _, H, W = real_images.shape
        device = real_images.device

        # Randomly sample some alpha between 0 and 1 for interpolation
        # where alpha is of the same shape for elementwise multiplication.
        alpha = torch.rand(N, 1)
        alpha = alpha.expand(N, int(real_images.nelement() / N)).contiguous()
        alpha = alpha.view(N, 4, H, W)
        alpha = alpha.to(device)

        # Obtain interpolates on line between real/fake images.
        interpolates = alpha * real_images.detach() \
            + ((1 - alpha) * fake_images.detach())
        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)

        # Get gradients of interpolates
        disc_interpolates = self.forward(interpolates)
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

class MyTrainer(mmc.training.Trainer):
    def _fetch_data(self, iter_dataloader):
        """
        Fetches the next set of data and refresh the iterator when it is exhausted.
        Follows python EAFP, so no iterator.hasNext() is used.
        """
        try:
            real_batch = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(self.dataloader)
            real_batch = next(iter_dataloader)

        #real_batch = (real_batch[0].to(self.device),
        #              real_batch[1].to(self.device))
	
        real_batch = (real_batch.to(self.device), None)

        return iter_dataloader, real_batch

#
