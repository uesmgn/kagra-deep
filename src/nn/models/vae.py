# Invariant Information Clustering for Unsupervised Image Classification and Segmentation
# https://arxiv.org/abs/1807.06653

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import Module, Gaussian, Reshape, ConvTranspose2dModule

__all__ = ["M1"]


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
            ConvTranspose2dModule(64, 3, 2, activation=None),
        )
        self.initialize_weights()

    def initialize_step(self):
        pass

    def forward(self, x, target):
        x, xt, z, z_mean, z_var = self.__forward(x, target)
        bce = self.__bce(x, xt)
        kl = self.__kl_norm(z_mean, z_var)
        loss = torch.cat([bce, kl])

        if self.training:
            return loss
        else:
            return loss, {"target": target, "z": z}

    def __forward(self, x, target):
        x_densed = self.encoder(x)
        z, z_mean, z_var = self.gaussian(x_densed)
        xt = self.decoder(z)
        return x, xt, z, z_mean, z_var

    def __bce(self, x, xt):
        bce = F.binary_cross_entropy_with_logits(xt, x, reduction="sum")
        return bce.unsqueeze(0)

    def __kl_norm(self, mean, var):
        kl = 0.5 * (torch.log(1.0 / var) + (var + torch.pow(mean, 2)) - 1).sum()
        return kl.unsqueeze(0)
