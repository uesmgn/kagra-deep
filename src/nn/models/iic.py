# Invariant Information Clustering for Unsupervised Image Classification and Segmentation
# https://arxiv.org/abs/1807.06653

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import Module

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

__all__ = ["IIC"]


class IIC(Module):
    def __init__(
        self,
        net,
        num_classes=10,
        num_classes_over=100,
        num_heads=5,
    ):
        super().__init__()
        # remove last fc layer
        self.encoder = nn.Sequential(*list(net.children())[:-1])
        self.num_heads = num_heads
        self.clustering_heads = nn.ModuleList(
            [nn.Linear(net.fc_in, num_classes) for _ in range(num_heads)]
        )
        self.over_clustering_heads = nn.ModuleList(
            [nn.Linear(net.fc_in, num_classes_over) for _ in range(num_heads)]
        )
        self.best_index = None
        self.initialize_weights()

    def initialize_step(self):
        print("initialize headers weights...")
        for m in self.clustering_heads:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.over_clustering_heads:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, *args):
        if self.training:
            if len(args) == 3:
                # unsupervised co-training loss
                x, xt, target = args
                return self.__usl(*args)
            elif len(args) == 6:
                # semi-supervised co-training loss
                return self.__ssl(*args)
            else:
                raise ValueError("args is invalid.")
        else:
            if len(args) == 2:
                # supervised training loss
                x, target = args
                y, _ = self.__forward(x)
                loss = self.__ce_heads(y, target, reduction="none")
                if self.best_index is None:
                    self.best_index = 0
                pred = torch.argmax(y[..., self.best_index], -1)
                return loss, target, pred
            else:
                raise ValueError("args is invalid.")

    def __usl(self, x, xt, _):
        y, y_over = self.__forward(x)
        yt, yt_over = self.__forward(xt)
        mi = self.__mi_heads(y, yt)
        mi_over = self.__mi_heads(y_over, yt_over)
        return torch.cat([mi, mi_over])

    def __sl(self, x, target):
        y, _ = self.__forward(x)
        return self.__ce_heads(y, target)

    def __ssl(self, lx, lxt, lt, ux, uxt, _):
        mi_labeled = self.__usl(lx, lxt, _)
        supervised = self.__sl(lx, lt)
        mi_unlabeled = self.__usl(ux, uxt, _)
        return torch.cat([mi_labeled, supervised, mi_unlabeled])

    def __forward(self, x):
        x_densed = self.encoder(x)
        y_heads = [F.softmax(head(x_densed), dim=-1) for head in self.clustering_heads]
        y = torch.stack(y_heads, -1)
        y_over_heads = [F.softmax(head(x_densed), dim=-1) for head in self.over_clustering_heads]
        y_over = torch.stack(y_over_heads, -1)
        return y, y_over

    def __mi(self, y, yt):
        eps = torch.finfo(y.dtype).eps
        _, k = y.size()
        p = (y.unsqueeze(2) * yt.unsqueeze(1)).sum(dim=0)
        p = ((p + p.t()) / 2) / p.sum()
        p[(p < eps).data] = eps
        pi = p.sum(dim=1).view(k, 1).expand(k, k)
        pj = p.sum(dim=0).view(1, k).expand(k, k)
        return (p * (torch.log(pi) + torch.log(pj) - torch.log(p))).sum()

    def __mi_heads(self, y, yt, reduction="sum"):
        loss = torch.stack([self.__mi(y[..., i], yt[..., i]) for i in range(self.num_heads)])
        if reduction == "sum":
            return loss.sum().unsqueeze(0)
        elif reduction == "none":
            return loss

    def __ce(self, y, target):
        return F.cross_entropy(y, target)

    def __ce_heads(self, y, target, reduction="sum"):
        loss = torch.stack([self.__ce(y[..., i], target) for i in range(self.num_heads)])
        if reduction == "sum":
            return loss.sum().unsqueeze(0)
        elif reduction == "none":
            return loss

    def __metrics(self, target, pred):

        target = target.cpu().numpy()
        pred = pred.cpu().numpy()

        precision_micro = precision_score(target, pred, average="micro")
        precision_macro = precision_score(target, pred, average="macro")
        precision_weighted = precision_score(target, pred, average="weighted")

        recall_micro = recall_score(target, pred, average="micro")
        recall_macro = recall_score(target, pred, average="macro")
        recall_weighted = recall_score(target, pred, average="weighted")

        f1_micro = f1_score(target, pred, average="micro", zero_division=0)
        f1_macro = f1_score(target, pred, average="macro", zero_division=0)
        f1_weighted = f1_score(target, pred, average="weighted", zero_division=0)

        cm = confusion_matrix(target, pred)

        params = {
            "precision_micro": precision_micro,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "recall_micro": recall_micro,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "cm": cm,
        }

        return params
