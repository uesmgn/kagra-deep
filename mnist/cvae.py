import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import abc


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super().__init__()
        self.x_conv = nn.Conv2d(
            in_channels, out_channels, stride=(1, 2), kernel_size=(1, 3), padding=(0, 1), bias=False
        )
        self.y_conv = nn.Conv2d(
            in_channels, out_channels, stride=(2, 1), kernel_size=(3, 1), padding=(1, 0), bias=False
        )
        self.block = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels),
        )
        if activation is None:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        identity = self.connection(x)
        x_conv = self.x_conv(x)
        y_conv = self.y_conv(x)
        x = self.block(y_conv @ x_conv) + identity
        return self.activation(x)


class TransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super().__init__()
        self.block = self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.connection = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
        )
        if activation is None:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        identity = self.connection(x)
        x = self.block(x) + identity
        return self.activation(x)


class Encoder(nn.Module):
    def __init__(self, ch_in=3, dim_out=1024):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(ch_in, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            Block(32, 64),
            Block(64, 96),
            Block(96, 128),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, dim_out, bias=False),
            nn.BatchNorm1d(dim_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, ch_in=3, dim_in=1024):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, 128 * 7 * 7, bias=False),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            TransposeBlock(128, 96),
            TransposeBlock(96, 64),
            TransposeBlock(64, 32),
        )
        self.fc = nn.Sequential(
            nn.ConvTranspose2d(32, ch_in, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.head(x)
        x = x.view(x.shape[0], 128, 7, 7)
        x = self.blocks(x)
        x = self.fc(x)
        return x


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


class Qz_x(nn.Module):
    def __init__(self, encoder, dim_z):
        super().__init__()
        self.encoder = encoder

        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gaussian = Gaussian(512, dim_z)

    def forward(self, x):
        x = self.encoder(x)
        logits = self.fc(x)
        z, mean, logvar = self.gaussian(logits)
        return z, mean, logvar


class Qz_xy(nn.Module):
    def __init__(self, encoder, dim_y, dim_z):
        super().__init__()
        self.encoder = encoder

        self.fc_x = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_y = nn.Sequential(
            nn.Linear(1024, dim_y, bias=False),
            nn.BatchNorm1d(dim_y),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gaussian = Gaussian(512 + dim_y, dim_z)

    def forward(self, x, y):
        x = self.encoder(x)
        x_logits = self.fc_x(x)
        y_logits = self.fc_y(x)
        z, mean, logvar = self.gaussian(torch.cat([x_logits, y_logits * y], dim=-1))
        return z, mean, logvar


class Qy_x(nn.Module):
    def __init__(self, encoder, dim_y):
        super().__init__()
        self.encoder = encoder

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
    def __init__(self, dim_y, dim_z):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(dim_y, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gaussian = Gaussian(1024, dim_z)

    def forward(self, y):
        y = self.fc(y)
        z, mean, logvar = self.gaussian(y)
        return z, mean, logvar


class Px_z(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z):
        x = self.decoder(z)
        return x


class M1(nn.Module):
    def __init__(
        self,
        ch_in=3,
        dim_z=64,
    ):
        super().__init__()
        self.encoder = Encoder(ch_in, 1024)
        self.decoder = Decoder(ch_in, dim_z)
        self.qz_x = Qz_x(self.encoder, dim_z)
        self.px_z = Px_z(self.decoder)
        self.weight_init()

    def load_state_dict_part(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            print(f"load state dict: {name}")
            own_state[name].copy_(param)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                try:
                    nn.init.zeros_(m.bias)
                except:
                    continue

    def forward(self, x):
        b = x.shape[0]
        z, mean, logvar = self.qz_x(x)
        x_recon = self.px_z(z)

        bce = self.bce(x, x_recon) / b
        kl_gauss = self.kl_gauss(mean, logvar, torch.zeros_like(mean), torch.ones_like(logvar)) / b
        return bce, kl_gauss

    def bce(self, x, x_recon):
        return F.binary_cross_entropy(x_recon, x, reduction="sum")

    def kl_gauss(self, mean_p, logvar_p, mean_q, logvar_q):
        return -0.5 * torch.sum(
            logvar_p - logvar_q + 1 - torch.pow(mean_p - mean_q, 2) / logvar_q.exp() - logvar_p.exp() / logvar_q.exp()
        )


class M2(nn.Module):
    def __init__(
        self,
        ch_in=3,
        dim_y=10,
        dim_z=64,
    ):
        super().__init__()
        self.encoder = Encoder(ch_in, 1024)
        self.decoder = Decoder(ch_in, dim_z)
        self.dim_y = dim_y
        self.qy_x = Qy_x(self.encoder, dim_y)
        self.qz_xy = Qz_xy(self.encoder, dim_y, dim_z)
        self.pz_y = Pz_y(dim_y, dim_z)
        self.px_z = Px_z(self.decoder)
        self.weight_init()

    def load_state_dict_part(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            print(f"load state dict: {name}")
            own_state[name].copy_(param)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                try:
                    nn.init.zeros_(m.bias)
                except:
                    continue

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
            kl_cat = self.kl_cat(qy_pi, F.softmax(torch.ones_like(qy_pi), dim=-1))
            return bce, kl_gauss, kl_cat
        else:
            # labeled learning
            ce = F.cross_entropy(qy_pi, y, reduction="sum")
            return bce, ce

    def bce(self, x, x_recon):
        return F.binary_cross_entropy(x_recon, x, reduction="sum")

    def kl_cat(self, q, p, eps=1e-8):
        return torch.sum(q * (torch.log(q + eps) - torch.log(p + eps)))

    def kl_gauss(self, mean_p, logvar_p, mean_q, logvar_q):
        return -0.5 * torch.sum(
            logvar_p - logvar_q + 1 - torch.pow(mean_p - mean_q, 2) / logvar_q.exp() - logvar_p.exp() / logvar_q.exp()
        )


class IIC(nn.Module):
    def __init__(self, ch_in, dim_y, dim_w):
        super().__init__()
        self.encoder = Encoder(ch_in, 1024)
        self.qy_x = Qy_x(self.encoder, dim_y)
        self.qw_x = Qy_x(self.encoder, dim_w)
        self.weight_init()

    def load_state_dict_part(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            print(f"load state dict: {name}")
            own_state[name].copy_(param)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                try:
                    nn.init.zeros_(m.bias)
                except:
                    continue

    def forward(self, x, v):
        # iic
        y_x, w_x = self.clustering(x)
        y_v, w_v = self.clustering(v)

        mi_y = self.mutual_info(y_x, y_v)
        mi_w = self.mutual_info(w_x, w_v)

        return mi_y, mi_w

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


class ClusterHeads(nn.Module):
    def __init__(self, dim_in, dim_y, dim_w):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim_in, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, dim_y),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim_in, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, dim_w),
        )

    def forward(self, x):
        y = F.softmax(self.fc1(x), -1)
        w = F.softmax(self.fc2(x), -1)
        return y, w


class M3(nn.Module):
    def __init__(
        self,
        ch_in=3,
        dim_y=10,
        dim_w=50,
        dim_z=64,
    ):
        super().__init__()
        self.qz_x = Qz_x(ch_in, dim_z)
        self.px_z = Px_z(ch_in, dim_z)
        self.cluster_heads = ClusterHeads(dim_z, dim_y, dim_w)
        self.weight_init()

    def load_state_dict_part(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            print(f"load state dict: {name}")
            own_state[name].copy_(param)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                try:
                    nn.init.zeros_(m.bias)
                except:
                    continue

    def forward(self, x):
        return self.vae(x)

    def params(self, x):
        _, z_x, _ = self.qz_x(x)
        y_x, w_x = self.cluster_heads(z_x)
        return z_x, y_x, w_x

    def vae(self, x):
        b = x.shape[0]
        qz, qz_mean, qz_logvar = self.qz_x(x)
        pz_mean, pz_logvar = torch.zeros_like(qz_mean), torch.ones_like(qz_logvar)
        x_recon = self.px_z(qz)
        bce = self.bce(x, x_recon) / b
        kl_gauss = self.kl_gauss(qz_mean, qz_logvar, pz_mean, pz_logvar) / b
        return bce, kl_gauss

    def iic(self, x, v, detach=False):
        b = x.shape[0]
        _, z_x, _ = self.qz_x(x)
        _, z_v, _ = self.qz_x(v)
        if detach:
            z_x, z_v = z_x.detach(), z_v.detach()
        y_x, w_x = self.cluster_heads(z_x)
        y_v, w_v = self.cluster_heads(z_v)
        mi_y = self.mutual_info(y_x, y_v) / b
        mi_w = self.mutual_info(w_x, w_v) / b
        return mi_y, mi_w

    def bce(self, x, x_recon):
        return F.binary_cross_entropy(x_recon, x, reduction="sum")

    def kl_gauss(self, mean_p, logvar_p, mean_q, logvar_q):
        return -0.5 * torch.sum(
            logvar_p - logvar_q + 1 - torch.pow(mean_p - mean_q, 2) / logvar_q.exp() - logvar_p.exp() / logvar_q.exp()
        )

    def mutual_info(self, x, y, alpha=2.0, eps=1e-8):
        p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(dim=0)
        p = ((p + p.t()) / 2) / p.sum()
        _, k = x.shape
        p[(p < eps).data] = eps
        pi = p.sum(dim=1).view(k, 1).expand(k, k)
        pj = p.sum(dim=0).view(1, k).expand(k, k)
        return (p * (alpha * torch.log(pi) + alpha * torch.log(pj) - torch.log(p))).sum()
