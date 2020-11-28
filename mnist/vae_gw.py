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
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        logits = self.fc(x)
        mean, logvar = torch.split(logits, logits.shape[-1] // 2, -1)
        x = self.reparameterize(mean, logvar)
        return x, mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        x = mean + eps * std
        return x


# inference net of y from x
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

    def forward(self, x, tau=0.5):
        x_encoded = self.encoder(x)
        logits = self.logits(x_encoded)
        y, pi = self.categorical_sample(logits)
        return y, pi

    def categorical_sample(self, logits, tau=0.5):
        y = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        pi = F.softmax(logits, -1)
        return y, pi


class Qz_xy(nn.Module):
    def __init__(
        self,
        encoder,
        dim_y,
        dim_z,
    ):
        super().__init__()
        self.encoder = encoder
        self.logits = nn.Sequential(
            nn.Linear(encoder.dim_out, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, dim_y),
        )
        self.gaussian = Gaussian(encoder.dim_out + dim_y, dim_z)

    def forward(self, x, y):
        x_encoded = self.encoder(x)
        logits = self.logits(x_encoded) * y
        z, z_mean, z_logvar = self.gaussian(torch.cat((x_encoded, logits), -1))
        return z, z_mean, z_logvar


class Qw_xz(nn.Module):
    def __init__(self, encoder, dim_w, dim_z):
        super().__init__()
        self.encoder = encoder
        self.logits = nn.Sequential(
            nn.Linear(encoder.dim_out + dim_z, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, dim_w),
        )

    def forward(self, x, z):
        x_encoded = self.encoder(x)
        logits = self.logits(torch.cat((x_encoded, z), -1))
        w, pi = self.categorical_sample(logits)
        return w, pi

    def categorical_sample(self, logits, tau=0.5):
        y = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        pi = F.softmax(logits, -1)
        return y, pi


class Pz_wy(nn.Module):
    def __init__(
        self,
        dim_w,
        dim_y,
        dim_z,
    ):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(dim_w + dim_y, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gaussian = Gaussian(1024, dim_z)

    def forward(self, w, y):
        logits = self.logits(torch.cat((x, z), -1))
        z, z_mean, z_logvar = self.gaussian(logits)
        return z, z_mean, z_logvar


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


def kl_cat(q, p, eps=1e-8):
    return torch.sum(q * (torch.log(q + eps) - torch.log(p + eps)))


def kl_gauss(mean_p, logvar_p, mean_q, logvar_q):
    return -0.5 * torch.sum(
        logvar_p
        - logvar_q
        + 1
        - torch.pow(mean_p - mean_q, 2) / logvar_q.exp()
        - logvar_p.exp() / logvar_q.exp()
    )


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
        dim_w=50,
        dim_z=64,
    ):
        super().__init__()

        encoder = Encoder(dim_x)
        decoder = Decoder(dim_z)

        self.qy_x = Qy_x(encoder, dim_y)
        self.qz_xy = Qz_xy(encoder, dim_x, dim_y)
        self.qw_xz = Qw_xz(encoder, dim_w, dim_z)
        self.pz_wy = Pz_wy(dim_w, dim_w, dim_y)
        self.px_z = Px_z(decoder, dim_z)

        self.weight_init()

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
        kl_gauss = kl_gauss(qz_xy_mean, qz_xy_logvar, qz_y_mean, qz_y_logvar) / b
        kl_cat = kl_cat(qz_xy_mean, qz_xy_logvar, qz_y_mean, qz_y_logvar) / b

        return bce, kld

    def params(self, x):
        y_logits, y_over_logits = self.cluster_iic(x)
        y, y_over = F.softmax(y_logits, -1), F.softmax(y_over_logits, -1)
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, y_over)
        return qz_xy_mean, y, y_over

    def mutual_info(self, x, y, alpha=1.0):
        x, y = F.softmax(x, -1), F.softmax(y, -1)
        eps = 1e-8
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
