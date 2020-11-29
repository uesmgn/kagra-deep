import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Encoder(nn.Module):
    def __init__(self, dim_encoded=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(6272, dim_encoded),
            nn.BatchNorm1d(dim_encoded),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super().__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.logits = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        logits = self.logits(x)
        mean, logvar = torch.split(logits, logits.shape[-1] // 2, -1)
        x = self._reparameterize(mean, logvar)
        return x, mean, logvar

    def _reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        x = mean + eps * std
        return x


class Decoder(nn.Module):
    def __init__(self, dim_z=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(dim_z, 6272),
            nn.BatchNorm1d(6272),
            nn.LeakyReLU(0.2),
            Reshape((32, 14, 14)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class Qz_xy(nn.Module):
    def __init__(
        self,
        dim_y=10,
        dim_z=2,
    ):
        super().__init__()
        self.encoder = Encoder(dim_encoded=1024)

        self.fc_x = nn.Sequential(
            nn.Linear(1024, 64),
        )
        self.fc_y = nn.Sequential(
            nn.Linear(1024, dim_y),
        )
        self.gaussian = Gaussian(64 + dim_y, dim_z)

    def forward(self, x, y):
        x = self.encoder(x)
        qx_logits = self.fc_x(x)
        qy_logits = self.fc_y(x)
        z, mean, logvar = self.gaussian(torch.cat([qx_logits, qy_logits * y], dim=-1))
        return z, mean, logvar


class Qy_x(nn.Module):
    def __init__(
        self,
        dim_y=10,
    ):
        super().__init__()
        self.encoder = Encoder(dim_encoded=1024)

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
        dim_y=10,
        dim_z=64,
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
        dim_z=64,
    ):
        super().__init__()

        self.decoder = Decoder(dim_z=dim_z)

    def forward(self, z):
        x = self.decoder(z)
        return x


class M2(nn.Module):
    def __init__(
        self,
        dim_y=10,
        dim_z=2,
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
