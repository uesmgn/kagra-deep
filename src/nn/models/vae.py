# Invariant Information Clustering for Unsupervised Image Classification and Segmentation
# https://arxiv.org/abs/1807.06653

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import Module, Gaussian, Reshape, ConvTranspose2dModule

__all__ = [
    'M1', 'M2'
]


class M1(Module):
    def __init__(self, net, z_dim=512):
        super().__init__()
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
            ConvTranspose2dModule(64, 3, 2, activation=None)
        )
        self.initialize_weights()

    def _bce(self, x, xt):
        bce = F.binary_cross_entropy_with_logits(xt, x, reduction='sum')
        return bce

    def _kl_norm(self, mean, var):
        kl = 0.5 * (torch.log(1. / var) + (var + torch.pow(mean, 2)) - 1).sum()
        return kl

    def criterion(self, x, xt, z_mean, z_var):
        bce = self._bce(x, xt)
        kl_norm = self._kl_norm(z_mean, z_var)
        return bce + kl_norm

    def forward(self, x, target=None):
        x_densed = self.encoder(x)
        z, z_mean, z_var = self.gaussian(x_densed)
        xt = self.decoder(z)
        params = dict(x=x, xt=xt, z=z, z_mean=z_mean, z_var=z_var)
        loss = self.criterion(x, xt, z_mean, z_var)
        return params, loss

class M2(Module):
    def __init__(self, net, z_dim=512):
        super().__init__()
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
            ConvTranspose2dModule(64, 3, 2, activation=None)
        )
        self.initialize_weights()

    def bce(self, x, xt):
        bce = F.binary_cross_entropy_with_logits(xt, x, reduction='none').view(x.shape[0], -1).sum(-1)
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
