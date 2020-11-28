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
    def __init__(self, dim_encoded=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        blocks = []
        blocks.append(ResBlock(64, 64))
        blocks.append(ResBlock(64, 64))

        blocks.append(ResBlock(64, 128, stride=2))
        blocks.append(ResBlock(128, 128))

        blocks.append(ResBlock(128, 256, stride=2))
        blocks.append(ResBlock(256, 256))

        blocks.append(ResBlock(256, 512, stride=2))
        blocks.append(ResBlock(512, 512))
        blocks.append(nn.Flatten()),
        blocks.append(nn.Linear(25088, dim_encoded))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
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
        self.blocks = nn.Sequential(
            nn.Linear(dim_z, 512 * 7 * 7),
            Reshape((512, 7, 7)),
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
        )

    def forward(self, x):
        return self.blocks(x)


class Qz_xy(nn.Module):
    def __init__(
        self,
        dim_x=64,
        dim_y=10,
        dim_z=2,
    ):
        super().__init__()
        self.encoder = Encoder(dim_encoded=1024)

        self.fc_x = nn.Sequential(
            nn.Linear(1024, dim_x),
        )
        self.fc_y = nn.Sequential(
            nn.Linear(1024, dim_y),
        )
        self.gaussian = Gaussian(dim_x + dim_y, dim_z)

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

    def forward(self, x, tau=0.5):
        x = self.encoder(x)
        logits = self.fc(x)
        qy = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
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
            nn.LeakyReLU(0.2, inplace=True),
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


def bce_with_logits(x, x_recon_logits):
    return F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction="sum")


def log_norm(x, mean, logvar):
    var = torch.exp(logvar) + 1e-8
    return -0.5 * (torch.log(2.0 * np.pi * var) + torch.pow(x - mean, 2) / var)


def log_norm_kl(x, mean, var, mean_, var_):
    log_p = log_norm(x, mean, var).sum(-1)
    log_q = log_norm(x, mean_, var_).sum(-1)
    return (log_p - log_q).sum()


def categorical_ce(pi, pi_prior):
    return -(pi * torch.log(pi_prior + 1e-8)).sum()


class LossDict(dict):
    def __init__(self, weights=None, **kwargs):
        self.weights = weights
        if weights is not None:
            if isinstance(weights, abc.Iterable):
                assert len(self) == len(weights)
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


class M4(nn.Module):
    def __init__(
        self,
        dim_x=64,
        dim_y=10,
        dim_y_over=50,
        dim_z=64,
    ):
        super().__init__()
        self.encoder = Encoder(dim_encoded=dim_x)
        self.cluster_head = nn.Sequential(
            nn.Linear(dim_x, dim_y),
        )
        self.cluster_head_over = nn.Sequential(
            nn.Linear(dim_x, dim_y_over),
        )
        self.qz_xy = Qz_xy(dim_x, dim_y_over, dim_z)
        self.pz_y = Pz_y(dim_y_over, dim_z)
        self.px_z = Px_z(dim_z)

        self.weight_init()

    def cluster_iic(self, x):
        x = self.encoder(x)
        y_logits = self.cluster_head(x)
        y_over_logits = self.cluster_head_over(x)
        return y_logits, y_over_logits

    def gumbel_sample(self, logits, tau=0.5):
        return F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)

    def load_state_dict_part(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            print("load state dict of {}.".format(name))
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
                nn.init.zeros_(m.bias)

    def forward(self, x, xt, mode="iic", weights=None, tau=0.5):
        y_logits, y_over_logits = self.cluster_iic(x)
        yt_logits, yt_over_logits = self.cluster_iic(xt)
        if mode == "iic":
            # iic
            mi = self.mutual_info(y_logits, yt_logits)
            mi_over = self.mutual_info(y_over_logits, yt_over_logits)
            params = dict(mi=mi, mi_over=mi_over)
        elif mode == "vae":
            # vae
            x_bce, x_kld = self.vae_loss(x, self.gumbel_sample(y_over_logits, tau=tau).detach())
            xt_bce, xt_kld = self.vae_loss(xt, self.gumbel_sample(y_over_logits, tau=tau).detach())
            params = dict(x_bce=x_bce, x_kld=x_kld, xt_bce=xt_bce, xt_kld=xt_kld)
        loss = LossDict(**params, weights=weights)
        return loss

    def vae_loss(self, x, y):
        b = x.shape[0]
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, y)
        qz_y, qz_y_mean, qz_y_logvar = self.pz_y(y)
        x_recon_logits = self.px_z(qz_xy)

        bce = bce_with_logits(x, x_recon_logits) / b
        kld = log_norm_kl(qz_xy, qz_xy_mean, qz_xy_logvar, qz_y_mean, qz_y_logvar) / b

        return bce, kld

    def params(self, x):
        y_logits, y_over_logits = self.cluster_iic(x)
        y, y_over = F.softmax(y_logits, -1), F.softmax(y_over_logits, -1)
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, y_over)
        return qz_xy_mean, y, y_over

    def mutual_info(self, x, y, alpha=1.0):
        x, y = F.softmax(x, -1), F.softmax(y, -1)
        eps = torch.finfo(x.dtype).eps
        b, k = x.shape
        p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(dim=0)
        p = ((p + p.t()) / 2) / p.sum()
        p[(p < eps).data] = eps
        pi = p.sum(dim=1).view(k, 1).expand(k, k)
        pj = p.sum(dim=0).view(1, k).expand(k, k)
        return (p * (alpha * torch.log(pi) + alpha * torch.log(pj) - torch.log(p))).sum() / b


# class M4(nn.Module):
#     def __init__(
#         self,
#         transform_fn,
#         augment_fn,
#         dim_x=64,
#         dim_y=10,
#         dim_y_over=50,
#         dim_z=64,
#     ):
#         super().__init__()
#         self.encoder = Encoder(dim_encoded=dim_x)
#         self.cluster_head = nn.Sequential(
#             nn.Linear(dim_x, dim_y),
#         )
#         self.cluster_head_over = nn.Sequential(
#             nn.Linear(dim_x, dim_y_over),
#         )
#         self.qz_xy = Qz_xy(dim_x, dim_y_over, dim_z)
#         self.pz_y = Pz_y(dim_y_over, dim_z)
#         self.px_z = Px_z(dim_z)
#
#         self.transform_fn = transform_fn
#         self.augment_fn = augment_fn
#
#         self.weight_init()
#
#     def cluster_iic(self, x):
#         x = self.encoder(x)
#         y_logits = self.cluster_head(x)
#         y_over_logits = self.cluster_head_over(x)
#         return y_logits, y_over_logits
#
#     def load_state_dict_part(self, state_dict):
#         own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name not in own_state:
#                 continue
#             print("load state dict of {}.".format(name))
#             own_state[name].copy_(param)
#
#     def weight_init(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_normal_(m.weight)
#             elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#                 nn.init.normal_(m.weight, mean=1, std=0.02)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 nn.init.zeros_(m.bias)
#
#     def forward(self, x, xt, beta=1.0, tau=0.5):
#         b = x.shape[0]
#
#         if y is None:
#             # unsupervised training
#
#             # iic
#             y_logits, y_over_logits = self.cluster_iic(x)
#             yt_logits, yt_over_logits = self.cluster_iic(xt)
#             mi = self.mutual_info(y_logits, yt_logits) / b
#             mi_over = self.mutual_info(y_over_logits, yt_over_logits) / b
#
#             # vae
#             qy = F.gumbel_softmax(y_over_logits, tau=tau, hard=False, dim=-1)
#             vae = self.vae(x, qy, beta) / b
#
#             total = mi + mi_over + vae
#
#             return total
#
#         else:
#             # supervised training
#
#             # vae
#             y_logits, y_over_logits = self.cluster_iic(x)
#             pi = F.softmax(y_logits, -1)
#             qy = F.gumbel_softmax(y_over_logits, tau=tau, hard=False, dim=-1)
#             vae = self.vae(x, qy, beta) / b
#             ce = F.cross_entropy(pi, torch.argmax(y, -1), reduction="sum") / b
#
#             total = vae + ce
#
#             return total
#
#     def vae(self, x, y_over, beta=1.0):
#         qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, y_over)
#         qz_y, qz_y_mean, qz_y_logvar = self.pz_y(y_over)
#         x_recon_logits = self.px_z(qz_xy)
#
#         bce = bce_with_logits(x, x_recon_logits)
#         kld = log_norm_kl(qz_xy, qz_xy_mean, qz_xy_logvar, qz_y_mean, qz_y_logvar)
#
#         return bce + beta * kld
#
#     def params(self, x):
#         y_logits, y_over_logits = self.cluster_iic(x)
#         y, y_over = F.softmax(y_logits, -1), F.softmax(y_over_logits, -1)
#         qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, y_over)
#         return qz_xy_mean, y, y_over
#
#     def mutual_info(self, x, y, alpha=1.0):
#         x, y = F.softmax(x, -1), F.softmax(y, -1)
#         eps = torch.finfo(x.dtype).eps
#         _, k = x.size()
#         p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(dim=0)
#         p = ((p + p.t()) / 2) / p.sum()
#         p[(p < eps).data] = eps
#         pi = p.sum(dim=1).view(k, 1).expand(k, k)
#         pj = p.sum(dim=0).view(1, k).expand(k, k)
#         return (p * (alpha * torch.log(pi) + alpha * torch.log(pj) - torch.log(p))).sum()
