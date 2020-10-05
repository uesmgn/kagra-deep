import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision

__all__ = [
    'Module', 'Reshape', 'Activation', 'Gaussian',
]

class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def load_part_of_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)

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

class Conv2dModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 batchnorm=True, activation=nn.ReLU(inplace=True)):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=stride+2, stride=stride,
                                padding=1, bias=False))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvTranspose2dModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 batchnorm=True, activation=nn.ReLU(inplace=True)):
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=stride+2, stride=stride,
                                         padding=1, bias=False))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

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
