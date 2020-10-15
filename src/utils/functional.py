import torch
from collections import abc
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import plotly.figure_factory as ff
import numpy as np

__all__ = ["to_device", "multi_class_metrics", "flatten", "tensordict"]


def multi_class_metrics(target, pred):
    target = target.view(-1).detach().cpu().numpy()
    pred = pred.view(-1).detach().cpu().numpy()

    precision_micro = precision_score(target, pred, average="micro", zero_division=0)
    precision_macro = precision_score(target, pred, average="macro", zero_division=0)
    precision_weighted = precision_score(target, pred, average="weighted", zero_division=0)

    recall_micro = recall_score(target, pred, average="micro", zero_division=0)
    recall_macro = recall_score(target, pred, average="macro", zero_division=0)
    recall_weighted = recall_score(target, pred, average="weighted", zero_division=0)

    f1_micro = f1_score(target, pred, average="micro", zero_division=0)
    f1_macro = f1_score(target, pred, average="macro", zero_division=0)
    f1_weighted = f1_score(target, pred, average="weighted", zero_division=0)

    labels = np.unique(target)
    cm = confusion_matrix(target, pred, labels=labels)

    fig = ff.create_annotated_heatmap(
        cm, x=labels, y=labels, annotation_text=cm, colorscale="Blues", showscale=True
    )
    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")

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
        "cm": fig,
    }
    return params


def to_device(device, *args):
    ret = []
    for arg in args:
        if torch.is_tensor(arg):
            ret.append(arg.to(device, non_blocking=True))
        elif isinstance(arg, abc.Sequence):
            ret.extend(to_device(device, *arg))
        else:
            raise ValueError(f"Input is invalid argument type: {type(arg)}.")
    return tuple(ret)


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class TensorDict(dict):
    def __init__(self):
        pass

    def cat(self, d):
        if isinstance(d, abc.MutableMapping):
            for key, x in d.items():
                assert torch.is_tensor(x)
                x = x.unsqueeze(0).detach().cpu()
                if key not in self:
                    self[key] = x
                else:
                    old = self[key]
                    self[key] = torch.cat([old, x], dim=0)
        else:
            raise ValueError("Invalid arguments.")

    def reduction(self, mode="mean", keep_dim=None, inplace=True):
        new = {}
        for key, x in self.items():
            indices = list(range(x.ndim))
            if isinstance(keep_dim, int):
                indices.pop(keep_dim)
                for i in indices:
                    if mode == "mean":
                        x = x.mean(i).unsqueeze(i)
                    elif mode == "sum":
                        x = x.sum(i).unsqueeze(i)
                    else:
                        raise ValueError("Invalid arguments.")
            else:
                if mode == "mean":
                    x = x.mean()
                elif mode == "sum":
                    x = x.sum()
                else:
                    raise ValueError("Invalid arguments.")
            new[key] = x.squeeze()
        if inplace:
            self.clear()
            self.update(new)
            return self
        return new

    def flatten(self, inplace=True, total=True):
        new = {}
        for key, value in self.items():
            if torch.is_tensor(value):
                try:
                    value = value.item()
                except:
                    value = value.tolist()
            if isinstance(value, abc.Sequence):
                if len(value) >= 10:
                    raise ValueError("dimention is too large.")
                tmp = 0
                for i, x in enumerate(value):
                    new_key = "{}_{}".format(key, i)
                    new[new_key] = x
                    tmp += x
                if total:
                    new["{}_total".format(key)] = tmp
            else:
                new[key] = value
        if inplace:
            self.clear()
            self.update(new)
            return self
        return new


def tensordict():
    return TensorDict()
