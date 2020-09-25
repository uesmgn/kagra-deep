import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = [
    'Module', 'Reshape', 'Activation', 'Gaussian',
]

class Module(nn.Module):
  def __init__(self):
    super().__init__()

  def initialize_weights(self, mode='fan_in'):
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
      elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super().__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

class Activation(nn.Module):
    def __init__(self, activation, **kwargs):
        super().__init__()
        activation = activation.lower()
        print(activation)
        if activation == 'relu':
            self.activation = nn.ReLU(**kwargs)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid(**kwargs)
        elif activation == 'softmax':
            self.activation = nn.Softmax(**kwargs)
        else:
            raise ValueError(f'activation {activation} is invalid.')

    def forward(self, x):
        return self.activation(x)

class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x, reparameterize=True):
        eps = torch.finfo(x.dtype).eps
        x_densed = self.dense(x)
        mean, logit = torch.split(x_densed, x_densed.shape[1] // 2, 1)
        var = F.softplus(logit) + eps
        if reparameterize:
            x = self._reparameterize(mean, var)
        else:
            x = mean
        return x, mean, var

    def _reparameterize(self, mean, var):
        if torch.is_tensor(var):
            std = torch.pow(var, 0.5)
        else:
            std = np.sqrt(var)
        eps = torch.randn_like(mean)
        x = mean + eps * std
        return x
