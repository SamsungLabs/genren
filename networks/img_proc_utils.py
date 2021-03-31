import torch, torch.nn as nn, torch.nn.functional as F
import math
from torchvision import transforms

class WeightedGradientImageDistance(nn.Module):
    def __init__(self):
        super(WeightedGradientImageDistance, self).__init__()
        self.gradient_estimator = ImageSpatialGradientCalculator()

    def forward(self, I1, I2, W): # In: B x nH x C x H x W, B x nH
        B, nH, C, H, W = I1.shape
        BnH = B * nH
        G1 = self.gradient_estimator.compute_gradient_magnitude(I1.reshape(BnH,C,H,W)).reshape(B,nH,H,W)
        G2 = self.gradient_estimator.compute_gradient_magnitude(I2.reshape(BnH,C,H,W)).reshape(B,nH,H,W)
        return ( (G1 - G2).pow(2).mean(-1).mean(-1) * W ).sum(-1).mean()

class GradientImageDistance(nn.Module):
    def __init__(self):
        super(GradientImageDistance, self).__init__()
        self.gradient_estimator = ImageSpatialGradientCalculator()

    def forward(self, I1, I2): # In: B x C x H x W
        G1 = self.gradient_estimator.compute_gradient_magnitude(I1)
        G2 = self.gradient_estimator.compute_gradient_magnitude(I2)
        return (G1 - G2).pow(2).mean()

class ImageSpatialGradientCalculator(nn.Module):
    """
    Computes spatial image gradients

    The original image min/max values are only needed if you want to call
        the `get_normalized_gradient_magnitude` method

    Args:
        orig_min (optional): minimum possible pixel value of an input image
        orig_max (optional): maximum possible pixel value of an input image
    """
    def __init__(self, orig_min=None, orig_max=None):
        super(ImageSpatialGradientCalculator, self).__init__()
        # Sobel filters
        xfilter = torch.tensor([[ 1.,  2.,   1.], 
                                [ 0.,  0.,   0.], 
                                [-1., -2.,  -1.]]).view(1,1,3,3)
        yfilter = torch.tensor([[1., 0., -1.], 
                                [2., 0., -2.], 
                                [1., 0., -1.]]).view(1,1,3,3)
        # Register filters, but not as parameters
        self.register_buffer('xfilter', xfilter)
        self.register_buffer('yfilter', yfilter)
        # Normalization variables
        if not orig_min is None: assert not orig_max is None
        if not orig_max is None: assert not orig_min is None
        self.omin = orig_min
        self.omax = orig_max
        if not self.omin is None: # 
            # See `get_normalized_gradient_magnitude` for details
            self.grad_range = abs(self.omax - self.omin)
            self.max_grad_size = 4.0 * self.grad_range
            self.norming_constant = math.sqrt(20.0) * self.grad_range
            #self.norming_constant = math.sqrt(2.0) * self.max_grad_size

    def forward(self, I):
        B, C, H, W = I.shape
        BC = B * C
        I = F.pad(I.view(BC,1,H,W), (1,1,1,1), mode='replicate')
        Gx = F.conv2d(I, self.xfilter, #.expand(BC, 1, -1, -1), 
                      bias=None, stride=1, padding=0, dilation=1, groups=1).view(B,C,H,W)
        Gy = F.conv2d(I, self.yfilter, #.expand(BC, 1, -1, -1), 
                      bias=None, stride=1, padding=0, dilation=1, groups=1).view(B,C,H,W)
        return (Gx, Gy)
   
    def get_normalized_gradient(self, I):
        gx, gy = self(I)
        return (Gx / self.max_grad_size, Gy / self.max_grad_size)

    def compute_gradient_magnitude(self, I, sqrt_eps=1e-6):
        """
        Computes the spatial image gradient as the per-channel average
            Euclidean norm of the spatial gradients.
        """
        Gx, Gy = self(I) # B x C=3 x H x W
        return torch.sqrt( (Gx**2 + Gy**2) + sqrt_eps ).mean(dim=1)

    def get_normalized_gradient_magnitude(self, I):
        """
        Computes normalized gradient magnitude image, using the extremal values of the 
            input images to ensure the output is in [0,1]. 
            
        The orig_min and orig_max arguments must have been passed to the constructor.
        
        Args:
            I: input images (B x C x H x W)

        Returns:
            G_n: the normalized image gradient magnitudes (each pixel value in [0,1])
        """
        mag = self.compute_gradient_magnitude(I) # B x H x W
        #return mag
        return mag / self.norming_constant

#---------------------------------------------------------------------------------------------#

##### Normalization methods #####

def minmax_normalize_images(images, into_minus1_1=False):
    """
    Normalizes a multichannel image (B x C x H x W) into [0,1] per channel via:
        k <- (k - min) / (max - min).

    Mins and maxes are computed *per image*.
    """
    eps = 1e-7
    minv = ( images.min(dim=-1)[0].min(dim=-1)[0] ).unsqueeze(-1).unsqueeze(-1) # B x C x 1 x 1
    maxv = ( images.max(dim=-1)[0].max(dim=-1)[0] ).unsqueeze(-1).unsqueeze(-1) # B x C x 1 x 1
    normed = (images - minv) / (maxv - minv)
    normed[ normed < 0.5 ] += eps
    normed[ normed > 0.5 ] -= eps
    if into_minus1_1:
        normed = (2.0 * normed) - 1.0
    return normed

class TensorNormalizer(nn.Module):
    def __init__(self, mean, std):
        super(TensorNormalizer, self).__init__()
        mu = torch.Tensor(mean).view(1,3,1,1)
        std = torch.Tensor(std).view(1,3,1,1)
        self.register_buffer('mu', mu)
        self.register_buffer('std', std)

    def forward(self, I):
        return (I - self.mu) / self.std


def denorm_std():
    """
    Returns a function that unnormalizes an image batch from [-1,1] to [0,1].
    I.e., reverses the standard (0.5,0.5,0.5), (0.5,0.5,0.5) pytorch/torchvision image normalizer
    Note: returns a class, which can be applied as a function.
    """
    return TensorNormalizer(mean=[-1,-1,-1], std=[2,2,2])

def norm_imnet():
    """
    Returns a function that performs imagenet-derived normalization on the input tensor.
    Note: returns a class, which can be applied as a function.
    """
    return TensorNormalizer(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

def denorm_imnet():
    """
    Returns a function that undoes the imagenet-based statistical normalization
    """
    return TensorNormalizer(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                std=[1/0.229, 1/0.224, 1/0.255])

def renorm_imnet_from_std():
    """
    Performs (1) denormalization from [-1,1] to [0,1], and (2) (re)normalization based on imagenet statistics
    """
    return nn.Sequential(denorm_std(), norm_imnet())

#---------------------------------------------------------------------------------------------#

##### Entropy #####

# Computes the pixelwise entropy as if each value were a Bernoulli parameter #

def pixelwise_entropy(S_I, eps=1e-3):
    """
    Computes the average per-pixel entropy of a scalar image with elements in [0,1].
    Each pixel is treated as the parameter of a Bernoulli RV.

    Args:
        S_I: scalar image (B x H x W)
        eps: real scalar, used for numerical stability

    Returns:
        Bernoulli pixelwise Shannon entropy, averaged over pixels and batch images
    """
    S_I = S_I.clamp(min=eps, max=1-eps)
    mI = 1.0 - S_I
    return -( S_I * torch.log(S_I) + mI * torch.log(mI) ).mean()





#

