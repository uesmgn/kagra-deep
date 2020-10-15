# Invariant Information Clustering for Unsupervised Image Classification and Segmentation
# https://arxiv.org/abs/1807.06653

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import Module, Gaussian, Classifier, Reshape, ConvTranspose2dModule

__all__ = ["M1", "M2"]


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


class M2(Module):
    def __init__(self, net, z_dim=512, num_classes=20):
        super().__init__()
        # remove last fc layer
        self.num_classes = num_classes
        self.encoder = nn.Sequential(*list(net.children())[:-1])
        self.classifier = Classifier(net.fc_in, num_classes)
        self.gaussian = Gaussian(net.fc_in + num_classes, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + num_classes, 512 * 7 * 7),
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

    def forward(self, *args):
        if self.training:
            lx, target, ux, _ = args
            sl_loss = self.__sl(lx, target)
            usl_loss = self.__usl(ux, _)
            return torch.cat([sl_loss, usl_loss])
        else:
            x, target = args
            x_densed = self.encoder(x)
            y = self.classifier(x_densed)
            z, z_mean, z_var = self.gaussian(torch.cat([x_densed, y]))
            pred = torch.argmax(y, -1)
            loss = self.__ce(y, target)
            return loss, {"target": target, "pred": pred, "z": z}

    def __usl(self, x, _):
        x_densed = self.encoder(x)
        y = self.classifier(x_densed)
        z, z_mean, z_var = self.gaussian(torch.cat([x_densed, y]))
        xt = self.decoder(torch.cat([z, y]))
        bce = self.__bce(x, xt)
        kl = self.__kl_norm(z_mean, z_var)
        loss = torch.cat([bce, kl])
        return loss

    def __sl(self, x, target):
        x_densed = self.encoder(x)
        y = self.classifier(x_densed)
        z, z_mean, z_var = self.gaussian(torch.cat([x_densed, y]))
        xt = self.decoder(torch.cat([z, y]))
        bce = self.__bce(x, xt)
        kl = self.__kl_norm(z_mean, z_var)
        ce = self.__ce(y, target)
        loss = torch.cat([bce, kl, ce])
        return loss

    def __ce(self, y, target):
        return F.cross_entropy(y, target).unsqueeze(0)

    def __bce(self, x, xt):
        bce = F.binary_cross_entropy_with_logits(xt, x, reduction="sum")
        return bce.unsqueeze(0)

    def __kl_norm(self, mean, var):
        kl = 0.5 * (torch.log(1.0 / var) + (var + torch.pow(mean, 2)) - 1).sum()
        return kl.unsqueeze(0)
