import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import abc


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, dim_out=512):
        super().__init__()
        self.dim_out = dim_out
        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            Block(64, 128, stride=2),
            Block(128, 256, stride=2),
            Block(256, 512, stride=2),
            nn.Flatten(),
        )
        self.fc = nn.Linear(25088, dim_out)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        return self.fc(x)


class TransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=None):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, in_channels, stride=stride, kernel_size=4, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if activation is None:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        x = self.block(x)
        return self.activation(x)


class Decoder(nn.Module):
    def __init__(self, dim_in=512):
        super().__init__()
        self.dim_in = dim_in
        self.head = nn.Sequential(
            nn.Linear(dim_in, 512 * 7 * 7),
            nn.BatchNorm1d(512 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            TransposeBlock(512, 256, stride=2),
            TransposeBlock(256, 128, stride=2),
            TransposeBlock(128, 64, stride=2),
            TransposeBlock(64, 3, stride=2, activation=nn.Sigmoid()),
        )

    def forward(self, x):
        x = self.head(x)
        x = x.view(x.shape[0], 512, 7, 7)
        return self.blocks(x)


class Gaussian(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mean, logvar):
        x = self.reparameterize(mean, logvar)
        return x, mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        x = mean + eps * std
        return x


class Qz_xy(nn.Module):
    def __init__(
        self,
        dim_y,
        dim_z,
    ):
        super().__init__()
        self.encoder = Encoder(1024)

        self.fc_x = nn.Sequential(
            nn.Linear(1024, 512),
        )
        self.fc_y = nn.Sequential(
            nn.Linear(1024, dim_y),
        )
        self.gaussian = Gaussian(512 + dim_y, dim_z)

    def forward(self, x, y):
        x = self.encoder(x)
        x_logits = self.fc_x(x)
        y_logits = self.fc_y(x)
        z, mean, logvar = self.gaussian(torch.cat([x_logits, y_logits * y], dim=-1))
        return z, mean, logvar


class Qy_x(nn.Module):
    def __init__(
        self,
        dim_y,
    ):
        super().__init__()
        self.encoder = Encoder(1024)

        self.fc = nn.Sequential(
            nn.Linear(1024, dim_y),
        )

    def forward(self, x, hard=False, tau=0.5):
        x = self.encoder(x)
        logits = self.fc(x)
        qy = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        pi = F.softmax(logits, dim=-1)
        return qy, pi


class Pz_y(nn.Module):
    def __init__(
        self,
        dim_y,
        dim_z,
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(dim_y, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.gaussian = Gaussian(1024, dim_z)

    def forward(self, y):
        y = self.fc(y)
        z, mean, logvar = self.gaussian(y)
        return z, mean, logvar


class Px_z(nn.Module):
    def __init__(
        self,
        dim_z,
    ):
        super().__init__()

        self.decoder = Decoder(dim_z)

    def forward(self, z):
        x = self.decoder(z)
        return x


class M2(nn.Module):
    def __init__(
        self,
        dim_y=10,
        dim_z=64,
    ):
        super().__init__()
        self.dim_y = dim_y
        self.qy_x = Qy_x(dim_y)
        self.qz_xy = Qz_xy(dim_y, dim_z)
        self.pz_y = Pz_y(dim_y, dim_z)
        self.px_z = Px_z(dim_z)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y=None):
        b = x.shape[0]
        qy, qy_pi = self.qy_x(x)
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, qy)
        qz_y, qz_y_mean, qz_y_logvar = self.pz_y(qy)
        x_recon = self.px_z(qz_xy)

        bce = self.bce(x, x_recon) / b

        if y is None:
            # unlabeled learning
            kl_gauss = self.kl_gauss(qz_xy_mean, qz_xy_logvar, qz_y_mean, qz_y_logvar) / b
            kl_cat = self.kl_cat(qy_pi, F.softmax(torch.ones_like(qy_pi), dim=-1)) / b
            return bce, kl_gauss, kl_cat
        else:
            # labeled learning
            ce = F.cross_entropy(qy_pi, y, reduction="sum") / b
            return bce, ce

    def bce(self, x, x_recon):
        return F.binary_cross_entropy(x_recon, x, reduction="sum")

    def kl_cat(self, q, p, eps=1e-8):
        return torch.sum(q * (torch.log(q + eps) - torch.log(p + eps)))

    def kl_gauss(self, mean_p, logvar_p, mean_q, logvar_q):
        return -0.5 * torch.sum(
            logvar_p
            - logvar_q
            + 1
            - torch.pow(mean_p - mean_q, 2) / logvar_q.exp()
            - logvar_p.exp() / logvar_q.exp()
        )
