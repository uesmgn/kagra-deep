# Invariant Information Clustering for Unsupervised Image Classification and Segmentation
# https://arxiv.org/abs/1807.06653

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import *

__all__ = [
    'IIC'
]

backbones = [
    'VGG11', 'VGG13', 'VGG16', 'VGG19',
    'ResNet18', 'ResNet34', 'ResNet50',
]

class IIC(Module):
    def __init__(self, backbone, in_channels=4, num_classes=10,
                 num_classes_over=100, num_heads=10):
        super().__init__()
        assert backbone in backbones
        net = globals()[backbone](in_channels=in_channels)
        # remove last fc layer
        self.encoder = nn.Sequential(*list(net.children())[:-1])
        self.num_heads = num_heads
        self.clustering_heads = nn.ModuleList([
            nn.Linear(net.fc_in, num_classes) for _ in range(num_heads)])
        self.over_clustering_heads = nn.ModuleList([
            nn.Linear(net.fc_in, num_classes_over) for _ in range(num_heads)])
        self.initialize_weights()

    def initialize_headers_weights(self):
        print('initialize headers weights...')
        for m in self.clustering_heads:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.over_clustering_heads:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, head_index=None):
        x_densed = self.encoder(x)
        y_outputs = [F.softmax(head(x_densed), dim=-1) for head in self.clustering_heads]
        y_over_outputs = [F.softmax(head(x_densed), dim=-1) for head in self.over_clustering_heads]
        return torch.stack(y_outputs), torch.stack(y_over_outputs)

    def crit(self, y, yt, head_dim=0):
        # get loss function and best head index
        eps = torch.finfo(y.dtype).eps
        y, yt = y.squeeze(head_dim), yt.squeeze(head_dim)

        def mutual_info(z, zt):
            _, k = z.size()
            p = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
            p = ((p + p.t()) / 2) / p.sum()
            p[(p < eps).data] = eps
            pi = p.sum(dim=1).view(k, 1).expand(k, k)
            pj = p.sum(dim=0).view(1, k).expand(k, k)
            return (p * (torch.log(pi) + torch.log(pj) - torch.log(p))).sum()

        if y.ndim == yt.ndim == 3:
            loss = []
            for i in range(y.shape[head_dim]):
                yi = torch.index_select(y, head_dim, torch.LongTensor([i])).squeeze()
                yti = torch.index_select(yt, head_dim, torch.LongTensor([i])).squeeze()
                loss.append(mutual_info(yi, yti))
            loss = torch.stack(loss)
            return loss
        elif y.ndim == yt.ndim == 2:
            loss = mutual_info(y, yt)
            return loss.unsqueeze(0)
        else:
            raise ValueError('ndim of y, yt must be 2 or 3.')
