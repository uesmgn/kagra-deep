# Invariant Information Clustering for Unsupervised Image Classification and Segmentation
# https://arxiv.org/abs/1807.06653

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multipledispatch import dispatch

from .. import nets

__all__ = ["M2"]


def categorical_cross_entropy(x):
    x_prior = F.softmax(torch.ones_like(x), dim=-1).detach()
    ce = -(x * torch.log(x_prior + 1e-8)).sum(-1)
    return ce


def labeled_elbo(x, x_recon_logits, y, z_mean, z_logvar):
    b, d = z_mean.shape
    bce = F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction="sum") / b
    kld = 0.5 * (z_logvar.exp() - z_logvar + torch.pow(z_mean, 2) - 1).sum() / b
    cat = categorical_cross_entropy(y).sum() / b  # constant, -np.log(1 / num_classes)
    l = bce + kld + cat
    return l


def unlabeled_elbo(x, x_recon_logits, y_logits, z_mean, z_logvar):
    b, d = z_mean.shape
    _, num_classes = y_logits.shape
    y_prob = torch.exp(y_logits)
    h = -(y_prob * y_logits).sum() / b

    bce = (
        F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction="none").view(b, -1).sum(-1)
    )
    kld = 0.5 * (z_logvar.exp() - z_logvar + torch.pow(z_mean, 2) - 1).sum(-1)
    cat = categorical_cross_entropy(F.one_hot(torch.tensor(0), num_classes=num_classes).float())
    l = (y_prob * (bce + kld + cat).unsqueeze(1)).sum() / b
    u = l + h
    return u


class M2_base(nets.BaseModule):
    def __init__(self, encoder_callback, decoder_callback, z_dim=512, num_classes=20):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            *list(encoder_callback(512).children()),  # drop last fc layer
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            *list(encoder_callback(num_classes).children()),
            nn.LogSoftmax(dim=-1),
        )

        self.gaussian = nets.Gaussian(512 + num_classes, z_dim)

        self.decoder = decoder_callback(z_dim + num_classes)

    @dispatch(object, object, object, object)
    def forward(self, lx, target, ux, _):
        if not self.training:
            raise NotImplementedError("The number of arguments is unexpected for evaluating.")
        labeled_loss = self.__labeled(lx, target)
        supervised_loss = self.__supervised(lx, target)
        unlabeled_loss = self.__unlabeled(ux)
        loss = torch.stack([labeled_loss, supervised_loss, unlabeled_loss])
        return loss

    @dispatch(object, object)
    def forward(self, x, target):
        if self.training:
            raise NotImplementedError("The number of arguments is unexpected for training.")
        x_densed = self.encoder(x)
        y_logits = self.classifier(x)
        y_prob = torch.exp(y_logits)
        y = torch.argmax(y_prob, -1)
        z, z_mean, z_logvar = self.gaussian(torch.cat([x_densed, y_prob], -1))
        loss = F.cross_entropy(y_logits, target)
        return {"y": y, "z": z, "loss": loss}

    def __labeled(self, x, target):
        x_densed = self.encoder(x)
        y = F.one_hot(target, num_classes=self.num_classes).float()
        z, z_mean, z_logvar = self.gaussian(torch.cat([x_densed, y], -1))
        x_recon_logits = self.decoder(torch.cat([z, y], -1))
        l = labeled_elbo(x, x_recon_logits, y, z_mean, z_logvar)
        return l

    def __supervised(self, x, target):
        y_logits = self.classifier(x)
        s = F.cross_entropy(y_logits, target)
        return s

    def __unlabeled(self, x):
        x_densed = self.encoder(x)
        y_logits = self.classifier(x)
        z, z_mean, z_logvar = self.gaussian(torch.cat([x_densed, torch.exp(y_logits)], -1))
        x_recon_logits = self.decoder(torch.cat([z, torch.exp(y_logits)], -1))
        u = unlabeled_elbo(x, x_recon_logits, y_logits, z_mean, z_logvar)
        return u


def M2(encoder="ResNet34", decoder="Decoder", z_dim=512, in_channels=3, num_classes=10):
    encoder_callback = lambda n: getattr(nets, encoder)(in_channels=in_channels, num_classes=n)
    decoder_callback = lambda n: getattr(nets, decoder)(in_channels=in_channels, num_classes=n)

    model = M2_base(
        encoder_callback,
        decoder_callback,
        z_dim=z_dim,
        num_classes=num_classes,
    )

    return model
