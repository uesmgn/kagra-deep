import inspect
import types
import torch
from collections import abc

__all__ = [
    "flatten", "get_net", "get_model", "get_optim",
    "get_dataset", "get_sampler", "get_loader"
]

def __class_keys(d):
    assert isinstance(d, types.ModuleType)
    keys = []
    for name, obj in inspect.getmembers(d):
        if inspect.isclass(obj) and name == obj.__name__:
            keys.append(name)
    return keys

def __instance(d, name, keys):
    if isinstance(d, types.ModuleType):
        d = vars(d)
    for key in keys:
        if key.lower() == name.lower():
            return d[key]
    raise ValueError("Available class names are {}, input {}.".format(keys, name))

def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_net(name, **params):
    from .nn import nets

    keys = __class_keys(nets)
    model = __instance(nets, name, keys)(**params)
    return model

def get_model(name, **params):
    from .nn import models

    keys = __class_keys(models)
    model = __instance(models, name, keys)(**params)
    return model

def get_optim(parameters, name, **params):
    if hasattr(torch.optim, name):
        return getattr(torch.optim, name)(parameters, **params)
    try:
        import apex
    except ImportError:
        print('Please install apex using...')
    if hasattr(apex.optimizers, name):
        return getattr(apex.optimizers, name)(parameters, **params)
    else:
        raise ValueError(f'optimizer {name} is invalid.')


def get_dataset(name, **params):
    from .data import datasets

    keys = __class_keys(datasets)
    dataset = __instance(datasets, name, keys)(**params)
    return dataset

def get_sampler(name, **params):
    from .data import samplers

    keys = __class_keys(samplers)
    sampler = __instance(samplers, name, keys)(**params)
    return sampler

def get_loader(dataset, **params):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, **params)
    return loader
