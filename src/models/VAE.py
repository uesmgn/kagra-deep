# Invariant Information Clustering for Unsupervised Image Classification and Segmentation
# https://arxiv.org/abs/1807.06653

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import *

__all__ = [
    'VAE'
]

backbones = [
    'VGG11', 'VGG13', 'VGG16', 'VGG19',
    'ResNet18', 'ResNet34', 'ResNet50',
]

def ConvTranspose2dModule(in_channels, out_channels, stride=1,
                          batchnorm=True, activation=nn.ReLU(inplace=True)):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=stride+2, stride=stride,
                                     padding=1, bias=False))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation)
    return nn.Sequential(*layers)

class VAE(Module):
    def __init__(self, backbone, in_channels=4, z_dim=512):
        super().__init__()
        assert backbone in backbones
        net = globals()[backbone](in_channels=in_channels)
        # remove last fc layer
        self.encoder = nn.Sequential(*list(net.children())[:-1])
        self.gaussian = Gaussian(net.fc_in, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Reshape((512, 7, 7)),
            ConvTranspose2dModule(512, 512, 2),
            ConvTranspose2dModule(512, 256, 2),
            ConvTranspose2dModule(256, 128, 2),
            ConvTranspose2dModule(128, 64, 2),
            ConvTranspose2dModule(64, in_channels, 2, activation=Activation('sigmoid'))
        )
        self.initialize_weights()

    def bce(self, x, xt):
        bce = F.binary_cross_entropy(xt, x, reduction='none').view(x.shape[0], -1).sum(-1)
        return bce

    def kl_norm(self, mean, var):
        mean_ = torch.zeros_like(mean)
        var_ = torch.ones_like(var)
        kl = 0.5 * ( torch.log(var_ / var) \
             + (var + torch.pow(mean - mean_, 2)) / var_ - 1).sum(-1)
        return kl

    def forward(self, x, reparameterize=True):
        x_densed = self.encoder(x)
        z, z_mean, z_var = self.gaussian(x_densed, reparameterize)
        xt = self.decoder(z)
        return xt, z, z_mean, z_var
