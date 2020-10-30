import numpy as np
import torch
import torch.nn as nn

__all__ = ["BaseModule", "Reshape", "Gaussian", "Decoder"]


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def load_state_dict_part(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)

    def initialize_weights(self, mode="fan_in"):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity="relu")
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


class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        mean = nn.Linear(in_dim, out_dim)
        logvar = nn.Linear(in_dim, out_dim)
        self.head = lambda x: (mean(x), logvar(x))

    def forward(self, x, reparameterize=True):
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


class Decoder(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Linear(num_classes, 512 * 7 * 7),
            Reshape((512, 7, 7)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return self.blocks(x)
