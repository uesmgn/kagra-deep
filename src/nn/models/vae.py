# Invariant Information Clustering for Unsupervised Image Classification and Segmentation
# https://arxiv.org/abs/1807.06653

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
            Reshape((512, 7, 7)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
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
        self.encoder = nn.Sequential(
            *list(net.children())[:-1],
            nn.Linear(net.fc_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            Classifier(512, num_classes),
        )
        self.gaussian = nn.Sequential(
            nn.Linear(512 + num_classes, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            Gaussian(512, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + num_classes, 512 * 7 * 7),
            Reshape((512, 7, 7)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
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
            labeled_loss = self.__labeled(lx, target)
            unlabeled_loss = self.__unlabeled(ux, _)
            return torch.cat([labeled_loss, unlabeled_loss])
        else:
            x, target = args
            x_densed = self.encoder(x)
            y, y_logits = self.classifier(x_densed)
            z, z_mean, z_var = self.gaussian(torch.cat([x_densed, y], -1))
            pred = torch.argmax(y_logits, -1)
            loss = self.__ce(y_logits, target)
            return loss, {"target": target, "pred": pred, "z": z}

    def __labeled(self, x, target):
        x_densed = self.encoder(x)
        y = F.one_hot(target, num_classes=self.num_classes).to(torch.float)
        z, z_mean, z_var = self.gaussian(torch.cat([x_densed, y], -1))
        xt = self.decoder(torch.cat([z, y], -1))
        log_p_x = self.__bce(x, xt)
        log_p_x = -np.log(1 / self.num_classes)
        log_p_z = self.__kl_norm(z_mean, z_var)
        labeled_loss = log_p_x + log_p_x + log_p_z

        _, y_logits = self.classifier(x_densed)
        sup_loss = self.__ce(y_logits, target)

        return torch.cat([labeled_loss, sup_loss])

    def __unlabeled(self, x, _):
        unlabeled_loss = 0
        x_densed = self.encoder(x)
        qy, _ = self.classifier(x_densed)
        y = F.one_hot(torch.arange(self.num_classes), num_classes=self.num_classes)
        y = y.to(x.device, dtype=x.dtype)
        for i in range(self.num_classes):
            qy_i = qy[:, i]
            y_i = y[:, i].repeat(x.shape[0], 1)
            z, z_mean, z_var = self.gaussian(torch.cat([x_densed, y_i], -1))
            xt = self.decoder(torch.cat([z, y_i], -1))

            log_p_x = self.__bce(x, xt)
            log_p_y = -np.log(1 / self.num_classes)
            log_p_z = self.__kl_norm(z_mean, z_var)
            log_q_y = torch.log(qy_i + 1e-8)
            unlabeled_loss += (log_p_x + log_p_y + log_p_z + log_q_y) * qy_i
        return unlabeled_loss

    def __ce(self, y, target):
        ce = F.cross_entropy(y, target)
        return ce.unsqueeze(0)

    def __bce(self, x, xt):
        bce = F.binary_cross_entropy_with_logits(xt, x, reduction="sum")
        return bce.unsqueeze(0)

    def __kl_norm(self, mean, var):
        kl = 0.5 * (torch.log(1.0 / var) + (var + torch.pow(mean, 2)) - 1).sum()
        return kl.unsqueeze(0)
