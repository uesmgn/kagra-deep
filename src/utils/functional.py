import torch
import numpy as np
from collections import abc
import re
import warnings
import matplotlib.colors as mc
import colorsys
import matplotlib.pyplot as plt

__all__ = ["cosine_similarity", "normalize", "pca", "cwm", "darken", "acronym", "to_device", "flatten", "tensordict"]


def cosine_similarity(x):
    x = x / x.norm(dim=-1)[:, None]
    return torch.mm(x, x.transpose(0, 1))


def normalize(x, axis=1):
    return (x - np.mean(x, axis=axis, keepdims=True)) / np.std(x, axis=axis, keepdims=True)


def pca(x, k, center=True):
    n = x.shape[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    h = torch.eye(n) - h
    h = h.to(x.device)
    x_center = torch.mm(h.double(), x.double())
    u, s, v = torch.svd(x_center)
    components = v[:k].t()
    return components


def cwm(i):
    colors = plt.cm.get_cmap("tab20").colors
    markers = "o^s*x"
    c = colors[i % 20]
    m = markers[int(i / 20)]
    return c, m


def darken(c, amount=0.5):
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def acronym(name):
    name = re.sub(
        r"(^[0-9a-zA-Z]{5,}(?=_))|((?<=_)[0-9a-zA-Z]*)",
        lambda m: str(m.group(1) or "")[:3] + str(m.group(2) or "")[:1],
        name,
    )
    name = name.replace("_", ".")
    return name


def getattr(d, name):
    assert isinstance(d, types.ModuleType)
    keys = []
    for key, obj in inspect.getmembers(d):
        if inspect.isclass(obj) and d.__name__ in obj.__module__:
            keys.append(key)
    for key in keys:
        if key.lower() == name.lower():
            return vars(d)[key]
    raise ValueError("Available class names are {}, but input is {}.".format(keys, name))


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
    def __init__(self, d=None):
        super().__init__()
        if isinstance(d, abc.MutableMapping):
            self.update(d)

    def cat(self, d, dim=0):
        if isinstance(d, abc.MutableMapping):
            for key, x in d.items():
                assert torch.is_tensor(x)
                x = x.detach().cpu()
                if key not in self:
                    self[key] = x
                else:
                    old = self[key]
                    self[key] = torch.cat([old, x], dim)
        else:
            raise ValueError("Invalid arguments.")

    def stack(self, d, dim=0):
        if isinstance(d, abc.MutableMapping):
            for key, x in d.items():
                assert torch.is_tensor(x)
                x = x.unsqueeze(dim).detach().cpu()
                if key not in self:
                    self[key] = x
                else:
                    old = self[key]
                    self[key] = torch.cat([old, x], dim)
        else:
            raise ValueError("Invalid arguments.")

    def mean(self, key, keep_dim=-1):
        x = self[key]
        if torch.is_tensor(x):
            if isinstance(keep_dim, int):
                dims = list(range(x.ndim))
                dims.pop(keep_dim)
                n = x.shape[keep_dim]
                x = x.permute(keep_dim, *dims).contiguous().view(n, -1).mean(-1)
            else:
                x = x.mean()
        self.update({key: x})
        return self

    def flatten(self, key):
        value = self[key]
        new = {}
        if torch.is_tensor(value):
            value = value.view(-1).tolist()
            if len(value) >= 10:
                warnings.warn("dimention is too large.")
                new[key] = value
            else:
                total = 0
                for i, x in enumerate(value):
                    new_key = "{}_{}".format(key, i)
                    new[new_key] = x
                    total += x
                new["{}_total".format(key)] = total
        else:
            new[key] = value

        self.pop(key)
        self.update(new)
        return self


def tensordict(d=None):
    return TensorDict(d)
