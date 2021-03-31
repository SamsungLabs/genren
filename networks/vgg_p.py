"""
From: https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py#L21
"""

from collections import namedtuple

import torch
from torchvision import models
from torchvision import transforms

#model = torchvision.models.resnet18()
#model.load_state_dict(torch.load('path/to/file'))

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, local_path=None):
        super(Vgg16, self).__init__()
        if local_path is None:
            vgg_pretrained_features = models.vgg16(pretrained=True).features
        else:
            _model = models.vgg16()
            _model.load_state_dict(torch.load(local_path))
            vgg_pretrained_features = _model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.intermed_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'] 

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", self.intermed_names)
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

    @staticmethod
    def expected_normalizer():
        """
        This normalizer is computed from ImageNet statistics.
        The network expects its inputs to be normalized with it.
        The inputs should be in [0,1].
        """
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
        return normalizer

    @staticmethod
    def imagenet_normalize_batch(batch, div_by_255=False):
        """
        By default, assumes the image is in [0,1].
        """
        # normalize using imagenet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        if div_by_255:
            batch = batch.div_(255.0)
        return (batch - mean) / std

#-------------------------------------------------------------------------------#

if __name__ == "__main__":

    f = Vgg16()

    B = 16
    imgs = torch.rand(B, 3, 32, 32)
    result = f(imgs)

    print('Sizes')
    for i, r in enumerate(result):
        print('\tR%d:' % i, r.shape)

    print('Done 1')

    n = Vgg16.expected_normalizer()
    nimgs = n(imgs)

    result2 = f(nimgs)

    print('Done 2')




#
