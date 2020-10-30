import torch
import torch.nn as nn
import numpy as np


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.logits = nn.Linear(in_dim, out_dim * 2)
        self.head = lambda x: torch.split(x, x.shape[1] // 2, 1)

    def forward(self, x, reparameterize=True):
        x = self.logits(x)
        mean, logvar = self.head(x)
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
