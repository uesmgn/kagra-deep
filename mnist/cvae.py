import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import abc


class ResBlock(nn.Module):
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
        )
        self.connection = None
        if in_channels != out_channels or stride != 1:
            self.connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x
        if self.connection is not None:
            identity = self.connection(x)
        x = self.block(x) + identity
        return self.activation(x)


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
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512),
            nn.Flatten(),
            nn.Linear(25088, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.head(x)
        return self.blocks(x)


class Decoder(nn.Module):
    def __init__(self, dim_in=512):
        super().__init__()
        self.dim_in = dim_in
        self.head = nn.Linear(dim_in, 512 * 7 * 7)
        self.blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
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


class Qy_x(nn.Module):
    def __init__(
        self,
        encoder,
        dim_y,
    ):
        super().__init__()
        self.encoder = encoder
        self.logits = nn.Sequential(
            nn.Linear(encoder.dim_out, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, dim_y),
        )

    def forward(self, x, hard=False, tau=0.5):
        x_encoded = self.encoder(x)
        logits = self.logits(x_encoded)
        pi = F.softmax(logits, -1)
        y = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        return y, pi


class Qz_xy(nn.Module):
    def __init__(
        self,
        encoder,
        dim_y,
        dim_z,
    ):
        super().__init__()
        self.encoder_x = encoder
        self.encoder_y = nn.Sequential(
            nn.Linear(dim_y, encoder.dim_out),
            nn.BatchNorm1d(encoder.dim_out),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(encoder.dim_out * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, dim_z * 2),
        )
        self.gaussian = Gaussian()

    def forward(self, x, y):
        logits = self.fc(torch.cat((self.encoder_x(x), self.encoder_y(y)), -1))
        z_mean, z_logvar = torch.split(logits, logits.shape[-1] // 2, -1)
        z, z_mean, z_logvar = self.gaussian(z_mean, z_logvar)
        return z, z_mean, z_logvar


class Pz_y(nn.Module):
    def __init__(
        self,
        dim_y,
        dim_z,
    ):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(dim_y, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, dim_z * 2),
        )
        self.gaussian = Gaussian()

    def forward(self, y):
        logits = self.logits(y)
        z_mean, z_logvar = torch.split(logits, logits.shape[-1] // 2, -1)
        z, z_mean, z_logvar = self.gaussian(z_mean, z_logvar)
        return z, z_mean, z_logvar


class Px_z(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z):
        x = self.decoder(z)
        return x


class LossDict(dict):
    def __init__(self, weights=None, **kwargs):
        self.weights = weights
        if weights is not None:
            if isinstance(weights, abc.Iterable):
                pass
            else:
                raise ValueError("weights must be NoneType or Iterable.")
        for k, v in kwargs.items():
            assert torch.is_tensor(v)
            self[k] = v

    @property
    def total(self):
        tmp = 0
        for i, (k, v) in enumerate(self.items()):
            try:
                tmp += v * self.weights[i]
            except:
                tmp += v
        return tmp

    def backward(self):
        self.total.backward()


class M2(nn.Module):
    def __init__(
        self,
        dim_y,
        dim_z,
    ):
        super().__init__()

        encoder = Encoder(1024)
        decoder = Decoder(dim_z)
        self.qy_x = Qy_x(encoder, dim_y)
        self.qz_xy = Qz_xy(encoder, dim_y, dim_z)
        self.pz_y = Pz_y(dim_y, dim_z)
        self.px_z = Px_z(decoder)
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

    def forward(self, x, weights=None, tau=0.5):
        # vae
        qy, qy_pi = self.qy_x(x)
        qz, qz_mean, qz_logvar = self.qz_xy(x, qy)
        pz, pz_mean, pz_logvar = self.pz_y(qy)
        px = self.px_z(pz)

        b = x.shape[0]
        bce = self.bce(x, px) / b
        klc = self.kl_cat(qy_pi, torch.ones_like(qy_pi) / qy_pi.shape[-1]) / b
        klg = self.kl_gauss(qz_mean, qz_logvar, pz_mean, pz_logvar) / b

        params = dict(bce=bce, klc=klc, klg=klg)
        loss = LossDict(**params, weights=weights)
        return loss

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


class IIC(nn.Module):
    def __init__(
        self,
        dim_y,
        dim_w,
    ):
        super().__init__()

        encoder = Encoder(1024)
        self.qy_x = Qy_x(encoder, dim_y)
        self.qw_x = Qy_x(encoder, dim_w)
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

    def forward(self, x, v, weights=None, tau=0.5):
        # iic
        y_x, w_x = self.clustering(x)
        y_v, w_v = self.clustering(v)

        b = x.shape[0]
        mi_y = self.mutual_info(y_x, y_v) / b
        mi_w = self.mutual_info(w_x, w_v) / b

        params = dict(mi_y=mi_y, mi_w=mi_w)
        loss = LossDict(**params, weights=weights)
        return loss

    def clustering(self, x):
        _, y_pi = self.qy_x(x)
        _, w_pi = self.qw_x(x)
        return y_pi, w_pi

    def mutual_info(self, x, y, alpha=2.0, eps=1e-8):
        p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(dim=0)
        p = ((p + p.t()) / 2) / p.sum()
        _, k = x.shape
        p[(p < eps).data] = eps
        pi = p.sum(dim=1).view(k, 1).expand(k, k)
        pj = p.sum(dim=0).view(1, k).expand(k, k)
        return (p * (alpha * torch.log(pi) + alpha * torch.log(pj) - torch.log(p))).sum()
