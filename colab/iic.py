import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, dim_encoded=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(6272, dim_encoded),
            nn.BatchNorm1d(dim_encoded),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class IIC(nn.Module):
    def __init__(
        self,
        dim_y=10,
        dim_w=20,
    ):
        super().__init__()
        self.encoder = Encoder(1024)
        self.cluster_y = nn.Sequential(
            nn.Linear(1024, dim_y),
        )
        self.cluster_w = nn.Sequential(
            nn.Linear(1024, dim_w),
        )

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

    def forward(self, x, v, alpha=1.0):
        qy_x, qw_x = self.clustering(x)
        qy_v, qw_v = self.clustering(v)

        mi_y = self.mutual_info(qy_x, qy_v, alpha)
        mi_w = self.mutual_info(qw_x, qw_v, alpha)

        return mi_y + mi_w

    def clustering(self, x):
        x = self.encoder(x)
        qy = F.softmax(self.cluster_y(x), -1)
        qw = F.softmax(self.cluster_w(x), -1)
        return qy, qw

    def mutual_info(self, x, y, alpha=1.0):
        eps = torch.finfo(x.dtype).eps
        b, k = x.shape
        p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(dim=0)
        p = ((p + p.t()) / 2) / p.sum()
        p[(p < eps).data] = eps
        pi = p.sum(dim=1).view(k, 1).expand(k, k)
        pj = p.sum(dim=0).view(1, k).expand(k, k)
        return (p * (alpha * torch.log(pi) + alpha * torch.log(pj) - torch.log(p))).sum() / b
