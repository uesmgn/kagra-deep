# Invariant Information Clustering for Unsupervised Image Classification and Segmentation
# https://arxiv.org/abs/1807.06653

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone as bb

backbones = [
    'VGG11', 'VGG13', 'VGG16', 'VGG19',
    'ResNet18', 'ResNet34', 'ResNet50',
]

def perturb_default(x, noise_rate=0.1):
    xt = x.clone()
    noise = torch.randn_like(x) * noise_rate
    xt += noise
    return xt

class IIC(nn.Module):
    def __init__(self, backbone, in_channels=4, num_classes=10,
                 num_classes_over=100, perturb_fn=None):
      super().__init__()
      assert backbone in backbones
      net = getattr(bb, backbone)(in_channels=in_channels)
      # remove last fc layer
      self.net = nn.Sequential(*list(net.children())[:-1])
      self.clustering = nn.Linear(net.fc_in, num_classes)
      self.over_clustering = nn.Linear(net.fc_in, num_classes_over)
      self.perturb_fn = perturb_fn
      if perturb_fn is None:
          self.perturb_fn = lambda x: perturb_default(x)

    def criterion(self, z, zt):
        _, k = z.size()
        p = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
        p = ((p + p.t()) / 2) / p.sum()
        eps = torch.finfo(p.dtype).eps
        p[(p < eps).data] = eps
        pi = p.sum(dim=1).view(k, 1).expand(k, k)
        pj = p.sum(dim=0).view(1, k).expand(k, k)
        return (p * (torch.log(pi) + torch.log(pj) - torch.log(p))).sum()

    def forward(self, x, perturb=False):
        if perturb:
            x = self.perturb_fn(x)
        x_densed = self.net(x)
        y = F.softmax(self.clustering(x_densed), dim=-1)
        y_over = F.softmax(self.over_clustering(x_densed), dim=-1)
        return y, y_over
