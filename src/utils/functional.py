import torch
from collections import abc
import warnings


__all__ = ["to_device", "flatten", "tensordict"]


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
