import torch
import torch.nn as nn
import numpy as np


class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.logits = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x, reparameterize=True):
        logits = self.logits(x)
        mean, logvar = torch.split(logits, logits.shape[-1] // 2, -1)
        if reparameterize:
            x = self._reparameterize(mean, logvar)
        else:
            x = mean
        return x, mean, logvar

    def _reparameterize(self, mean, logvar):
        if torch.is_tensor(logvar):
            std = torch.exp(0.5 * logvar)
        else:
            std = np.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        x = mean + eps * std
        return x


class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super().__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)
