import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import models
from ..layers import Reshape, Gaussian

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


def unlabeled_elbo(x, x_recon_logits, y_prob, z_mean, z_logvar):
    b, d = z_mean.shape
    _, num_classes = y_prob.shape
    h = -(y_prob * y_logits).sum() / b

    bce = (
        F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction="none").view(b, -1).sum(-1)
    )
    kld = 0.5 * (z_logvar.exp() - z_logvar + torch.pow(z_mean, 2) - 1).sum(-1)
    cat = categorical_cross_entropy(F.one_hot(torch.tensor(0), num_classes=num_classes).float())
    l = (y_prob * (bce + kld + cat).unsqueeze(1)).sum() / b
    u = l + h
    return u


class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=10):
        super().__init__()
        net = models.VGG16(in_channels=in_channels, num_classes=out_dim)
        self.blocks = nn.Sequential(
            *list(net.children()),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class Classifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        net = models.VGG16(in_channels=in_channels, num_classes=num_classes)
        self.blocks = nn.Sequential(
            *list(net.children()),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_dim=10, out_channels=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Linear(in_dim, 512 * 7 * 7),
            Reshape((512, 7, 7)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        return self.blocks(x)


class M2(pl.LightningModule):
    def __init__(self, in_channels=3, mid_dim=512, z_dim=512, num_classes=10):
        super().__init__()

        self.num_classes = num_classes
        self.encoder = Encoder(in_channels=in_channels, out_dim=mid_dim)
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)
        self.gaussian = Gaussian(mid_dim + num_classes, z_dim)
        self.decoder = Decoder(in_dim=z_dim + num_classes, out_channels=in_channels)

    def forward(self, x):
        x_dense = self.encoder(x)
        y_logits = self.classifier(x)
        y_prob = F.softmax(y_logits, dim=-1)
        z, z_mean, z_logvar = self.gaussian(torch.cat([x_dense, y_prob]))
        x_recon_logits = self.decoder(torch.cat([z, y_prob]))
        return x_recon_logits

    def training_step(self, batch, batch_idx):
        (lx, ly), (ux, _) = batch
        print(lx.device)
        print(ly.device)
        print(ux.device)
        ly = F.one_hot(ly, num_classes=self.num_classes)
        labeled_loss = self.__labeled_loss(lx, ly)
        supervised_loss = self.__supervised_loss(lx, ly)
        unlabeled_loss = self.__unlabeled_loss(ux)
        self.log("train_loss", loss)
        return loss

    def __labeled_loss(self, x, y):
        x_densed = self.encoder(x)
        z, z_mean, z_logvar = self.gaussian(torch.cat([x_densed, y], -1))
        x_recon_logits = self.decoder(torch.cat([z, y], -1))
        loss = labeled_elbo(x, x_recon_logits, y, z_mean, z_logvar)
        return loss

    def __supervised_loss(self, x, y):
        y_logits = self.classifier(x)
        loss = (-y * F.log_softmax(y_logits, dim=-1)).sum(-1)
        return loss

    def __unlabeled_loss(self, x):
        x_densed = self.encoder(x)
        y_logits = self.classifier(x)
        y_prob = F.softmax(y_logits, dim=-1)
        z, z_mean, z_logvar = self.gaussian(torch.cat([x_densed, y_prob], -1))
        x_recon_logits = self.decoder(torch.cat([z, y_prob], -1))
        loss = unlabeled_elbo(x, x_recon_logits, y_prob, z_mean, z_logvar)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
