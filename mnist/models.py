import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import abc


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=None):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.connection = None
        if in_channels != out_channels or stride != 1:
            self.connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        if activation is None:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        identity = x
        if self.connection is not None:
            identity = self.connection(x)
        x = self.block(x) + identity
        return self.activation(x)


class ResBlockDec(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super().__init__()
        self.block = self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, stride=2, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.connection = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
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
    def __init__(self, ch_in=3, dim_out=512):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(ch_in, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, dim_out, bias=False),
            nn.BatchNorm1d(dim_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, ch_in=3, dim_in=1024):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, 256 * 7 * 7, bias=False),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ResBlockDec(256, 128),
            ResBlockDec(128, 64),
            ResBlockDec(64, 32),
            ResBlockDec(32, ch_in, activation=nn.Sigmoid()),
        )

    def forward(self, x):
        x = self.head(x)
        x = x.view(x.shape[0], 256, 7, 7)
        x = self.blocks(x)
        return x


class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.logits = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x, l=1):
        logits = self.logits(x)
        mean, logvar = torch.split(logits, logits.shape[-1] // 2, -1)
        tmp = []
        for i in range(l):
            x = self._reparameterize(mean, logvar)
            tmp.append(x)
        return torch.stack(tmp, 0).squeeze(), mean, logvar

    def _reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        x = mean + eps * std
        return x


class Qz_x(nn.Module):
    def __init__(self, encoder, dim_z=512):
        super().__init__()
        self.encoder = encoder

        self.fc = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gaussian = Gaussian(1024, dim_z)

    def forward(self, x, l=1):
        x = self.encoder(x)
        logits = self.fc(x)
        z, mean, logvar = self.gaussian(logits, l)
        return z, mean, logvar


class Qz_xy(nn.Module):
    def __init__(self, encoder, dim_y=100, dim_z=512):
        super().__init__()
        self.encoder = encoder

        self.fc_x = nn.Sequential(
            nn.Linear(1024, dim_y, bias=False),
            nn.BatchNorm1d(dim_y),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_y = nn.Sequential(
            nn.Linear(1024, dim_y, bias=False),
            nn.BatchNorm1d(dim_y),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gaussian = Gaussian(dim_y + dim_y, dim_z)

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

    def forward(self, x, tau=0.5):
        hard = not self.training
        x = self.encoder(x)
        logits = self.fc(x)
        qy = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        pi = F.softmax(logits, dim=-1)
        return qy, pi, logits


class Qy_z(nn.Module):
    def __init__(self, dim_y, dim_z):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(dim_z, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, dim_y, bias=False),
        )

    def forward(self, x, tau=0.5):
        logits = self.fc(x)
        pi = F.softmax(logits, dim=-1)
        return pi


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


class AE(nn.Module):
    def __init__(
        self,
        ch_in=3,
        dim_z=512,
    ):
        super().__init__()
        self.encoder = Encoder(ch_in, dim_z)
        self.decoder = Decoder(ch_in, dim_z)
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
        z = self.encoder(x)
        x_ = self.decoder(z)
        bce = self.bce(x, x_) / b
        return bce

    def get_params(self, x):
        z = self.encoder(x)
        return z

    def bce(self, x, x_recon):
        return F.binary_cross_entropy(x_recon, x, reduction="sum")


class VAE(nn.Module):
    def __init__(
        self,
        ch_in=3,
        dim_z=512,
    ):
        super().__init__()
        self.encoder = Encoder(ch_in, 1024)
        self.mean = nn.Sequential(
            nn.Linear(1024, dim_z, bias=False),
            nn.BatchNorm1d(dim_z),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.logvar = nn.Sequential(
            nn.Linear(1024, dim_z, bias=False),
            nn.BatchNorm1d(dim_z),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = Decoder(ch_in, dim_z)
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

    def forward(self, x, l=10):
        b = x.shape[0]
        h = self.encoder(x)
        z_mean, z_logvar = self.mean(h), self.logvar(h)
        z = self.reparameterize(z_mean, z_logvar)
        x_ = self.decoder(z)
        bce = self.bce(x, x_) / b
        kl_gauss = self.kl_gauss(z_mean, z_logvar, torch.zeros_like(z_mean), torch.ones_like(z_logvar)) / b
        return bce, kl_gauss

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        x = mean + eps * std
        return x

    def get_params(self, x):
        h = self.encoder(x)
        z_mean = self.mean(h)
        return z_mean

    def bce(self, x, x_recon):
        return F.binary_cross_entropy(x_recon, x, reduction="sum")

    def kl_gauss(self, mean_p, logvar_p, mean_q, logvar_q):
        return -0.5 * torch.sum(logvar_p - logvar_q + 1 - torch.pow(mean_p - mean_q, 2) / logvar_q.exp() - logvar_p.exp() / logvar_q.exp())


class IIC(nn.Module):
    def __init__(self, ch_in, dim_w=30, dim_w_over=100, dim_z=512, num_heads=10):
        super().__init__()
        self.use_multi_heads = num_heads > 1
        self.num_heads = num_heads
        self.encoder = Encoder(ch_in, 1024)
        self.mean = nn.Sequential(
            nn.Linear(1024, dim_z, bias=False),
            nn.BatchNorm1d(dim_z),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if self.use_multi_heads:
            self.classifier = nn.ModuleList([self.gen_classifier(dim_z, dim_w) for _ in range(self.num_heads)])
        else:
            self.classifier = self.gen_classifier(dim_z, dim_w)

        if self.use_multi_heads:
            self.over_classifier = nn.ModuleList([self.gen_classifier(dim_z, dim_w_over) for _ in range(self.num_heads)])
        else:
            self.over_classifier = self.gen_classifier(dim_z, dim_w_over)

        self.weight_init()

    def gen_classifier(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, dim_out),
        )

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

    def forward(self, x, y, lam=1.0, detach=True):
        z_x, z_y = self.encoder(x), self.encoder(y)
        z_x, z_y = self.mean(z_x), self.mean(z_y)
        if detach:
            z_x, z_y = z_x.detach(), z_y.detach()
        w_v, w_u = self.clustering(z_x), self.clustering(z_y)
        w_v_over, w_u_over = self.over_clustering(z_x), self.over_clustering(z_y)
        mi = self.mutual_info(w_v, w_u, lam=lam)
        mi_over = self.mutual_info(w_v_over, w_u_over, lam=lam)
        return mi, mi_over

    def get_params(self, x):
        assert not self.training
        z = self.encoder(x)
        z = self.mean(z)
        w = self.clustering(z)
        w_over = self.over_clustering(z)
        return z, w, w_over

    def clustering(self, x):
        if self.use_multi_heads:
            tmp = []
            for classifier in self.classifier:
                w = F.softmax(classifier(x), dim=-1)
                tmp.append(w)
            return torch.stack(tmp, dim=-1)
        else:
            w = F.softmax(self.classifier(x), dim=-1)
            return w

    def over_clustering(self, x):
        if self.use_multi_heads:
            tmp = []
            for classifier in self.over_classifier:
                w = F.softmax(classifier(x), dim=-1)
                tmp.append(w)
            return torch.stack(tmp, dim=-1)
        else:
            w = F.softmax(self.over_classifier(x), dim=-1)
            return w

    def mutual_info(self, x, y, lam=1.0, eps=1e-8):
        if x.ndim == 2:
            m = 1
            p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(0)
            p = ((p + p.t()) / 2) / p.sum()
            _, k = x.shape
            p[(p < eps).data] = eps
            pi = p.sum(dim=1).view(k, 1).expand(k, k).pow(lam)
            pj = p.sum(dim=0).view(1, k).expand(k, k).pow(lam)
        elif x.ndim == 3:
            p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(0)
            p = ((p + p.permute(1, 0, 2)) / 2) / p.sum()
            p[(p < eps).data] = eps
            _, k, m = x.shape
            pi = p.sum(dim=1).view(k, -1).expand(k, k, m).pow(lam)
            pj = p.sum(dim=0).view(k, -1).expand(k, k, m).pow(lam)
        return (p * (torch.log(pi) + torch.log(pj) - torch.log(p))).sum() / m


class IIC_VAE(nn.Module):
    def __init__(self, ch_in, dim_w=30, dim_w_over=100, dim_z=512, num_heads=10):
        super().__init__()
        self.use_multi_heads = num_heads > 1
        self.num_heads = num_heads
        self.encoder = Encoder(ch_in, 1024)
        self.mean = nn.Sequential(
            nn.Linear(1024, dim_z, bias=False),
            nn.BatchNorm1d(dim_z),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.logvar = nn.Sequential(
            nn.Linear(1024, dim_z, bias=False),
            nn.BatchNorm1d(dim_z),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if self.use_multi_heads:
            self.classifier = nn.ModuleList([self.gen_classifier(dim_z, dim_w) for _ in range(self.num_heads)])
        else:
            self.classifier = self.gen_classifier(dim_z, dim_w)

        if self.use_multi_heads:
            self.over_classifier = nn.ModuleList([self.gen_classifier(dim_z, dim_w_over) for _ in range(self.num_heads)])
        else:
            self.over_classifier = self.gen_classifier(dim_z, dim_w_over)

        self.weight_init()

    def gen_classifier(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, dim_out),
        )

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

    def forward(self, x, lam=1.0):
        h = self.encoder(x)
        z_mean, z_logvar = self.mean(h), self.logvar(h)
        z = self.reparameterize(z_mean, z_logvar.detach())
        w_v, w_u = self.clustering(z_mean), self.clustering(z)
        w_v_over, w_u_over = self.over_clustering(z_mean), self.over_clustering(z)
        mi = self.mutual_info(w_v, w_u, lam=lam)
        mi_over = self.mutual_info(w_v_over, w_u_over, lam=lam)
        return mi, mi_over

    def get_params(self, x):
        assert not self.training
        h = self.encoder(x)
        z_mean = self.mean(h)
        w = self.clustering(z_mean)
        w_over = self.over_clustering(z_mean)
        return z, w, w_over

    def clustering(self, x):
        if self.use_multi_heads:
            tmp = []
            for classifier in self.classifier:
                w = F.softmax(classifier(x), dim=-1)
                tmp.append(w)
            return torch.stack(tmp, dim=-1)
        else:
            w = F.softmax(self.classifier(x), dim=-1)
            return w

    def over_clustering(self, x):
        if self.use_multi_heads:
            tmp = []
            for classifier in self.over_classifier:
                w = F.softmax(classifier(x), dim=-1)
                tmp.append(w)
            return torch.stack(tmp, dim=-1)
        else:
            w = F.softmax(self.over_classifier(x), dim=-1)
            return w

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        x = mean + eps * std
        return x

    def mutual_info(self, x, y, lam=1.0, eps=1e-8):
        if x.ndim == 2:
            m = 1
            p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(0)
            p = ((p + p.t()) / 2) / p.sum()
            _, k = x.shape
            p[(p < eps).data] = eps
            pi = p.sum(dim=1).view(k, 1).expand(k, k).pow(lam)
            pj = p.sum(dim=0).view(1, k).expand(k, k).pow(lam)
        elif x.ndim == 3:
            p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(0)
            p = ((p + p.permute(1, 0, 2)) / 2) / p.sum()
            p[(p < eps).data] = eps
            _, k, m = x.shape
            pi = p.sum(dim=1).view(k, -1).expand(k, k, m).pow(lam)
            pj = p.sum(dim=0).view(k, -1).expand(k, k, m).pow(lam)
        return (p * (torch.log(pi) + torch.log(pj) - torch.log(p))).sum() / m
