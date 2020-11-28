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
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


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


class M1(nn.Module):
    def __init__(
        self,
        dim_x=10,
        dim_y=10,
        dim_z=2,
    ):
        super().__init__()

        self.qy_x = Qy_x(dim_y)
        self.qz_xy = Qz_xy(dim_x, dim_y, dim_z)
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

    def forward(self, x, beta=10.0):
        b = x.shape[0]
        qy, pi = self.qy_x(x)
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, qy)
        qz_y, qz_y_mean, qz_y_logvar = self.pz_y(qy)
        x_recon_logits = self.px_z(qz_xy)

        bce = bce_with_logits(x, x_recon_logits) / b
        kld = log_norm_kl(qz_xy, qz_xy_mean, qz_xy_logvar, qz_y_mean, qz_y_logvar) / b
        ce = categorical_ce(pi, F.softmax(torch.ones_like(pi), dim=-1).detach()) / b

        return bce + beta * kld + ce

    def features_from_x(self, x):
        qy, pi = self.qy_x(x)
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, qy)
        if self.training:
            return qz_xy
        else:
            return qz_xy_mean

    def features_from_y(self, y):
        qz_y, qz_y_mean, qz_y_logvar = self.pz_y(y)
        return qz_y


class M2(nn.Module):
    def __init__(
        self,
        dim_x=10,
        dim_y=10,
        dim_z=2,
    ):
        super().__init__()

        self.qy_x = Qy_x(dim_y)
        self.qz_xy = Qz_xy(dim_x, dim_y, dim_z)
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

    def forward(self, ux, lx, y, alpha=1000.0, beta=10.0):
        loss_unlabeled, losses_unlabeled = self.loss_unlabeled(ux, beta)
        loss_labeled, losses_labeled = self.loss_labeled(lx, y, alpha)
        losses_unlabeled.update(losses_labeled)
        return loss_unlabeled + loss_labeled, losses_unlabeled

    def loss_unlabeled(self, x, beta=5.0):
        b = x.shape[0]
        qy, pi = self.qy_x(x)
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, qy)
        qz_y, qz_y_mean, qz_y_logvar = self.pz_y(qy)
        x_recon_logits = self.px_z(qz_xy)

        bce = bce_with_logits(x, x_recon_logits) / b
        kld = log_norm_kl(qz_xy, qz_xy_mean, qz_xy_logvar, qz_y_mean, qz_y_logvar) / b
        ce = categorical_ce(pi, F.softmax(torch.ones_like(pi), dim=-1).detach()) / b

        return bce + beta * kld + ce, {"bce_u": bce, "kld_u": kld, "ce_u": ce}

    def loss_labeled(self, x, y, alpha=1000.0):
        b = x.shape[0]
        qy, pi = self.qy_x(x)
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, y)
        qz_y, qz_y_mean, qz_y_logvar = self.pz_y(qy)
        x_recon_logits = self.px_z(qz_xy)

        bce = bce_with_logits(x, x_recon_logits) / b
        kld = log_norm_kl(qz_xy, qz_xy_mean, qz_xy_logvar, qz_y_mean, qz_y_logvar) / b
        ce = F.cross_entropy(pi, torch.argmax(y, dim=-1), reduction="sum") / b

        return 0.1 * (bce + kld) + alpha * ce, {"bce_l": bce, "kld_l": kld, "ce_l": ce}

    def features_from_x(self, x):
        qy, pi = self.qy_x(x)
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, qy)
        if self.training:
            return qz_xy
        else:
            return qz_xy_mean

    def features_from_y(self, y):
        qz_y, qz_y_mean, qz_y_logvar = self.pz_y(y)
        return qz_y


class M3(nn.Module):
    def __init__(
        self,
        transform_fn,
        dim_x=64,
        dim_y=10,
        dim_y_over=15,
        dim_z=2,
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

        self.transform_fn = transform_fn

        self.weight_init()

    def cluster_iic(self, x):
        x = self.encoder(x)
        y_logits = self.cluster_head(x)
        y = F.softmax(y_logits, -1)
        y_over = F.softmax(self.cluster_head_over(x), -1)
        return y, y_over, y_logits

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

    def forward(self, x, target=None, alpha=10.0, beta=10.0, gamma=100.0, tau=0.5):
        b = x.shape[0]
        # iic
        xt = self.transform_fn(x)
        y, y_over, y_pi = self.cluster_iic(x)
        yt, yt_over, yt_pi = self.cluster_iic(xt)
        mi = self.mutual_info(y, yt) / b
        mi_over = self.mutual_info(y_over, yt_over) / b

        # vae
        qy_over = F.gumbel_softmax(y_over, tau=tau, hard=False, dim=-1)
        vae = self.vae(x, qy_over) / b

        total = alpha * mi + beta * mi_over + vae

        if target is not None:
            target = torch.argmax(target, dim=-1)
            y_ce = F.cross_entropy(y_pi, target, reduction="sum") / b
            yt_ce = F.cross_entropy(yt_pi, target, reduction="sum") / b
            total += gamma * (y_ce + yt_ce)

        return total, {"mi": mi, "mi_over": mi_over, "vae": vae}

    def vae(self, x, y):
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, y)
        qz_y, qz_y_mean, qz_y_logvar = self.pz_y(y)
        x_recon_logits = self.px_z(qz_xy)

        bce = bce_with_logits(x, x_recon_logits)
        kld = log_norm_kl(qz_xy, qz_xy_mean, qz_xy_logvar, qz_y_mean, qz_y_logvar)

        return bce + kld

    def params(self, x):
        qy, qy_over, _ = self.cluster_iic(x)
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, qy_over)
        return qz_xy_mean, qy, qy_over

    def mutual_info(self, x, y, alpha=1.0):
        eps = torch.finfo(x.dtype).eps
        _, k = x.size()
        p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(dim=0)
        p = ((p + p.t()) / 2) / p.sum()
        p[(p < eps).data] = eps
        pi = p.sum(dim=1).view(k, 1).expand(k, k)
        pj = p.sum(dim=0).view(1, k).expand(k, k)
        return (p * (alpha * torch.log(pi) + alpha * torch.log(pj) - torch.log(p))).sum()


class M4(nn.Module):
    def __init__(
        self,
        transform_fn=None,
        dim_x=64,
        dim_y=10,
        dim_y_over=50,
        dim_z=2,
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

        self.transform_fn = transform_fn

        self.weight_init()

    def cluster_iic(self, x):
        x = self.encoder(x)
        y_logits = self.cluster_head(x)
        y_over_logits = self.cluster_head_over(x)
        return y_logits, y_over_logits

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

    def forward(self, x, y=None, beta=1.0, tau=0.5):
        b = x.shape[0]

        if y is None:
            # unsupervised training

            # iic
            xt = self.transform_fn(x)
            y_logits, y_over_logits = self.cluster_iic(x)
            yt_logits, yt_over_logits = self.cluster_iic(xt)
            mi = self.mutual_info(y_logits, yt_logits) / b
            mi_over = self.mutual_info(y_over_logits, yt_over_logits) / b

            # vae
            qy = F.gumbel_softmax(y_over_logits, tau=tau, hard=False, dim=-1)
            vae = self.vae(x, qy, beta) / b

            total = mi + mi_over + vae

            return total

        else:
            # supervised training

            # vae
            y_logits, y_over_logits = self.cluster_iic(x)
            pi = F.softmax(y_logits, -1)
            qy = F.gumbel_softmax(y_over_logits, tau=tau, hard=False, dim=-1)
            vae = self.vae(x, qy, beta) / b
            ce = F.cross_entropy(pi, torch.argmax(y, -1), reduction="sum") / b

            total = vae + ce

            return total

    def vae(self, x, y_over, beta=1.0):
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, y_over)
        qz_y, qz_y_mean, qz_y_logvar = self.pz_y(y_over)
        x_recon_logits = self.px_z(qz_xy)

        bce = bce_with_logits(x, x_recon_logits)
        kld = log_norm_kl(qz_xy, qz_xy_mean, qz_xy_logvar, qz_y_mean, qz_y_logvar)

        return bce + beta * kld

    def params(self, x):
        y_logits, y_over_logits = self.cluster_iic(x)
        y, y_over = F.softmax(y_logits, -1), F.softmax(y_over_logits, -1)
        qz_xy, qz_xy_mean, qz_xy_logvar = self.qz_xy(x, y_over)
        return qz_xy_mean, y, y_over

    def mutual_info(self, x, y, alpha=1.0):
        x, y = F.softmax(x, -1), F.softmax(y, -1)
        eps = torch.finfo(x.dtype).eps
        _, k = x.size()
        p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(dim=0)
        p = ((p + p.t()) / 2) / p.sum()
        p[(p < eps).data] = eps
        pi = p.sum(dim=1).view(k, 1).expand(k, k)
        pj = p.sum(dim=0).view(1, k).expand(k, k)
        return (p * (alpha * torch.log(pi) + alpha * torch.log(pj) - torch.log(p))).sum()


class Classifier(nn.Module):
    def __init__(
        self,
        dim_x=64,
        dim_y=10,
    ):
        super().__init__()
        self.encoder = Encoder(dim_encoded=dim_x)
        self.cluster_head = nn.Sequential(
            nn.Linear(dim_x, dim_y),
        )

    def forward(self, x, target):
        b = x.shape[0]
        x = self.encoder(x)
        y_logits = self.cluster_head(x)
        y = F.softmax(y_logits, -1)
        ce = F.cross_entropy(y_logits, target, reduction="sum") / b
        return ce

    def classify(self, x):
        x = self.encoder(x)
        y_logits = self.cluster_head(x)
        y = F.softmax(y_logits, -1)
        return y
