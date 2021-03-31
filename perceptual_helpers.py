"""
Useful methods for perceptual losses and Gram matrix texture statistics computations.

See: Perceptual Losses for Real-Time Style Transfer, Johnson et al, 2016 [1]

Following [1], we use MSE losses by default.
"""

import torch, torch.nn as nn, torch.nn.functional as F

from networks.vgg_p import Vgg16
#from cnns.block_helpers import LinBnAct, Reshaper, Symmetrizer
from networks.img_proc_utils import TensorNormalizer
#from utils.img_proc_utils import renorm_imnet_from_std

class PerceptualMethodsHandler(nn.Module):
    """
    Provides methods for 
        (1) computing Gram matrices and
        (2) calculating perceptual losses.
    """
    def __init__(self, use_denorm_renorm=True, vgg_path=None): #, perceptual_loss_weights=None, gram_matrix_loss_weights=None):
        super(PerceptualMethodsHandler, self).__init__()
        # Initialize VGG-16 network used for perceptual losses and appearance statistics
        # It has been pretrained on ImageNet
        self.vgg_network = Vgg16(requires_grad=False, local_path=vgg_path)

        # We unnormalize from [-1,1] --> [0,1].
        # Then we renormalize using ImageNet statistics.
        # This is so that the VGG gets stats that it expects.
        self.use_denorm_renorm = use_denorm_renorm
        self.normalization_corrector = renorm_imnet_from_std()

        # Loss term weights, if we want to weight different layers differently
        #if perceptual_loss_weights is None:
        #    self.wp = [1.0, 1.0, 1.0, 1.0]
        #else:
        #    self.wp = perceptual_loss_weights

    def vgg(self, I):
        if self.use_denorm_renorm: 
            I = self.normalization_corrector(I)
        return self.vgg_network(I)

    def features_and_gramians(self, I):
        """
        Extract perceptual features and Gramians from input image batch

        Args:
            I: images

        Returns:
            (F, M) = (Feature maps, Gramians)
        """
        Fs = self.vgg(I)
        Ms = [ gram_matrix(F) for F in Fs ]
        return (Fs, Ms)

    def perceptual_loss_from_feats(self, F1, F2):
        """
        Computes the perceptual loss between two sets of feature maps
        """
        L = 0.0
        for i in range(len(F1)):
            L += (F1[i] - F2[i]).pow(2).mean()
        return L

    def weighted_perceptual_loss_from_feats(self, F1, F2, W, s):
        L = 0.0
        B, nH, C, height, width = s
        for i in range(len(F1)):
            BnH, c, h, w = F1[i].shape
            L += ( W *  #                   W            H            C
                    ( F1[i].view(B,nH,c,h,w) 
                        - 
                      F2[i].view(B,nH,c,h,w)
                    ).pow(2).mean(dim=-1).mean(dim=-1).sum(dim=-1) 
                 ).sum(dim=-1).mean()
        return L

    def perceptual_loss(self, I1, I2):
        """
        Computes the perceptual loss between two image batches I1 and I2 via VGG16
        """
        return self.perceptual_loss_from_feats(self.vgg(I1), self.vgg(I2))

    def weighted_perceptual_loss(self, I1, I2, weights):
        """
        Computes the perceptual loss between two image batches I1 and I2 via VGG16,
            weighted by hypotheses.

        I1, I2: B x nH x C x H x W
        weights: B x nH
        """
        B, nH, C, H, W = I1.shape
        def clean(x):
            alpha = x[:,:,3,:,:].unsqueeze(2)
            return (x[:, :, 0:3, :, :] * alpha)
        if I1.shape[2] == 4:
            I1 = clean(I1)
        if I2.shape[2] == 4:
            I2 = clean(I2)
        C = 3 # now, after cleaning
        return self.weighted_perceptual_loss_from_feats(
                        self.vgg(I1.view(B*nH, C, H, W)), #.view(B,nH,C,H,W), 
                        self.vgg(I2.view(B*nH, C, H, W)), #.view(B,nH,C,H,W), 
                        weights, I1.shape)

    def extract_Gram_statistics(self, I):
        """
        Computes the texture statistics of the input images

        Args:
            I: images (B x C x H x W)

        Returns:
            the Gram matrices at each intermediate layer of the VGG network
        """
        Fs = self.vgg(I)
        gram_style = [ gram_matrix(F) for F in Fs ]
        return gram_style

    def compute_Gram_matrix_loss(self, M1, M2):
        """
        Computes a loss (L1) between two texture statistics (Gram matrix) sets.

        Args:
            M1: first set of Gram matrices
            M2: second set of Gram matrices
        """
        L = 0.0
        for m1, m2 in zip(M1, M2):
            L += torch.abs(m1 - m2).mean()
        return L
    
    def compute_Gram_matrix_loss_mse(self, M1, M2):
        """
        Computes a loss (MSE) between two texture statistics (Gram matrix) sets.

        Args:
            M1: first set of Gram matrices
            M2: second set of Gram matrices
        """
        L = 0.0
        for m1, m2 in zip(M1, M2):
            L += (m1 - m2).pow(2).mean()
        return L
    
    def compute_texture_statistics_loss(self, I1, I2, return_gram_matrices=False):
        """
        Computes the difference in texture statistics directly from two image 
            minibatches, by computing their Gram matrices and calculating 
            a metric upon them.
        """
        M1 = self.extract_Gram_statistics(I1)
        M2 = self.extract_Gram_statistics(I2)
        L = self.compute_Gram_matrix_loss(M1, M2)
        if return_gram_matrices: return L, (M1, M2)
        return L

    def expected_features_shape(self, S):
        """
        Returns the expected shape of the VGG16 feature maps, given an
            input of size (length = width = height) S.

        Args:
            S: width/height of the input image
        """
        return [ [64, S, S], [128, S//2, S//2], [256, S//4, S//4], [512, S//8, S//8] ]

    def expected_gram_style_shape(self):
        """
        Returns the expected shapes of the Gram matrices (no dependence on image size).
        """
        return [ [64, 64], [128, 128], [256, 256], [512, 512] ]

    def get_gramian_predictor(self, latent_dim):
        """
        Calling this *constructs* a predictor model
        """
        return GramianPredictor(latent_dim, self.expected_gram_style_shape())
   
def gram_matrix(y):
    """
    From:
        https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def renorm_imnet_from_std():
    """
    Performs (1) denormalization from [-1,1] to [0,1], and (2) (re)normalization based on imagenet statistics
    """
    return nn.Sequential(denorm_std(), norm_imnet())

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

"""
It's worth noting the following quote from Johnson et al, regarding the bias introduced by the use of an ImageNet trained VGG:

    In these results it is clear that the trained style transfer network is aware of
    the semantic content of images. For example in the beach image in Figure 7 the
    people are clearly recognizable in the transformed image but the background is
    warped beyond recognition; similarly in the cat image, the cat’s face is clear in
    the transformed image, but its body is not. One explanation is that the VGG-16
    loss network has features which are selective for people and animals since these
    objects are present in the classification dataset on which it was trained. Our
    style transfer networks are trained to preserve VGG-16 features, and in doing so
    they learn to preserve people and animals more than background objects
"""



#
