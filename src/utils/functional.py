import torch
from collections import abc

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
    def __init__(self):
        pass

    def stack(self, d):
        for k, v in d.items():
            assert torch.is_tensor(v)
            if v.is_cuda:
                v = v.cpu()
            if k not in self:
                self[k] = v
            else:
                new = torch.cat([self[k], v])
                self[k] = new


def tensordict():
    return TensorDict()
