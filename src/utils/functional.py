import torch
from collections import abc
import re
import warnings
import matplotlib.colors as mc
import colorsys

__all__ = ["colormap", "darken", "acronym", "to_device", "flatten", "tensordict"]


def colormap(i):
    colors = [
        "#5A5156",
        "#E4E1E3",
        "#F6222E",
        "#FE00FA",
        "#16FF32",
        "#3283FE",
        "#FEAF16",
        "#B00068",
        "#1CFFCE",
        "#90AD1C",
        "#2ED9FF",
        "#DEA0FD",
        "#AA0DFE",
        "#F8A19F",
        "#325A9B",
        "#C4451C",
        "#1C8356",
        "#85660D",
        "#B10DA1",
        "#FBE426",
        "#1CBE4F",
        "#FA0087",
        "#FC1CBF",
        "#F7E1A0",
        "#C075A6",
        "#782AB6",
        "#AAF400",
        "#BDCDFF",
        "#822E1C",
        "#B5EFB5",
        "#7ED7D1",
        "#1C7F93",
        "#D85FF7",
        "#683B79",
        "#66B0FF",
        "#3B00FB",
    ]

    i = i % len(colors)
    return colors[i]


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
