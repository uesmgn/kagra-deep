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
        transform_fn,
        dim_encoded=64,
        n_classes=10,
        n_classes_over=20,
    ):
        super().__init__()
        self.encoder = Encoder(dim_encoded=dim_encoded)
        self.cluster_head = nn.Sequential(
            nn.Linear(dim_encoded, n_classes),
        )
        self.cluster_head_over = nn.Sequential(
            nn.Linear(dim_encoded, n_classes_over),
        )
        self.transform_fn = transform_fn

        self.weight_init()

    def weight_init(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
        for m in self.cluster_head:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.cluster_head_over:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, alpha=1.0):
        xt = self.transform_fn(x)

        y, y_over = self.iic_clustering(x)
        yt, yt_over = self.iic_clustering(xt)

        iic_loss = self.mutual_info(y, yt, alpha) + self.mutual_info(y_over, yt_over, alpha)

        return iic_loss

    def cluster(self, x):
        y, y_over = self.iic_clustering(x)
        return y, y_over

    def iic_clustering(self, x):
        h = self.encoder(x)
        y = F.softmax(self.cluster_head(h), -1)
        y_over = F.softmax(self.cluster_head_over(h), -1)
        return y, y_over

    def mutual_info(self, x, y, alpha=1.0):
        eps = torch.finfo(x.dtype).eps
        _, k = x.size()
        p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(dim=0)
        p = ((p + p.t()) / 2) / p.sum()
        p[(p < eps).data] = eps
        pi = p.sum(dim=1).view(k, 1).expand(k, k)
        pj = p.sum(dim=0).view(1, k).expand(k, k)
        return (p * (alpha * torch.log(pi) + alpha * torch.log(pj) - torch.log(p))).sum()
