# Invariant Information Clustering for Unsupervised Image Classification and Segmentation
# https://arxiv.org/abs/1807.06653

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import *

__all__ = [
    'IIC_VAE'
]

backbones = [
    'VGG11', 'VGG13', 'VGG16', 'VGG19',
    'ResNet18', 'ResNet34', 'ResNet50',
]

def perturb_default(x, noise_rate=0.1):
    xt = x.clone()
    noise = torch.randn_like(x) * noise_rate
    xt += noise
    return xt

def ConvTranspose2dModule(in_channels, out_channels, stride=1,
                          batchnorm=True, activation=nn.ReLU(inplace=True)):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=stride+2, stride=stride,
                                     padding=1, bias=False))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation)
    return nn.Sequential(*layers)

class IIC_VAE(Module):
    def __init__(self, backbone, in_channels=4, num_classes=10,
                 num_classes_over=100, z_dim=512, num_heads=10, perturb_fn=None):
        super().__init__()
        assert backbone in backbones
        net = globals()[backbone](in_channels=in_channels)
        # remove last fc layer
        self.encoder = nn.Sequential(*list(net.children())[:-1])
        self.gaussian = Gaussian(net.fc_in, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Reshape((512, 7, 7)),
            ConvTranspose2dModule(512, 512, 2),
            ConvTranspose2dModule(512, 256, 2),
            ConvTranspose2dModule(256, 128, 2),
            ConvTranspose2dModule(128, 64, 2),
            ConvTranspose2dModule(64, in_channels, 2, activation=Activation('sigmoid'))
        )
        self.clustering_heads = nn.ModuleList([
            nn.Linear(net.fc_in, num_classes) for _ in range(num_heads)])
        self.over_clustering_heads = nn.ModuleList([
            nn.Linear(net.fc_in, num_classes_over) for _ in range(num_heads)])
        self.perturb_fn = perturb_fn
        if perturb_fn is None:
            self.perturb_fn = lambda x: perturb_default(x)
        self.initialize_weights()

    def initialize_headers_weights(self):
        for m in self.clustering_heads:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.over_clustering_heads:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def iic_loss(self, z, zt):
        _, k = z.size()
        p = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
        p = ((p + p.t()) / 2) / p.sum()
        eps = torch.finfo(p.dtype).eps
        p[(p < eps).data] = eps
        pi = p.sum(dim=1).view(k, 1).expand(k, k)
        pj = p.sum(dim=0).view(1, k).expand(k, k)
        return (p * (torch.log(pi) + torch.log(pj) - torch.log(p))).sum()

    def vae_loss(self, x, x_generated, z_mean, z_var):
        bce = F.binary_cross_entropy(x_generated, x, reduction='none').view(x.shape[0], -1)
        mean_ = torch.zeros_like(z_mean)
        var_ = torch.ones_like(z_var)
        kl = 0.5 * ( torch.log(var_ / z_var) \
               + (z_var + torch.pow(z_mean - mean_, 2)) / var_ - 1)
        return (bce.sum(-1) + kl.sum(-1)).mean()

    def forward(self, x, perturb=False):
        if perturb:
            x = self.perturb_fn(x)
        x_densed = self.encoder(x)
        return x_densed

    def iic(self, x_densed, head_index=None):
        if isinstance(head_index, int):
            y_output = F.softmax(self.clustering_heads[head_index](x_densed), dim=-1)
            y_over_output = F.softmax(self.over_clustering_heads[head_index](x_densed), dim=-1)
            return y_output, y_over_output
        else:
            y_outputs = [F.softmax(head(x_densed), dim=-1) for head in self.clustering_heads]
            y_over_outputs = [F.softmax(head(x_densed), dim=-1) for head in self.over_clustering_heads]
            return y_outputs, y_over_outputs

    def vae(self, x_densed, reparameterize=True):
        z, z_mean, z_var = self.gaussian(x_densed, reparameterize)
        x_generated = self.decoder(z)
        return x_generated, z, z_mean, z_var
