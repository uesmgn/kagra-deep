import torch
import inspect
from collections import abc

__all__ = [
    'flatten', 'get_optim'
]

def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_optim(parameters, cfg):
    try:
        import apex
    except ImportError:
        print('Please install apex using...')
    if hasattr(torch.optim, cfg.name):
        cls = getattr(torch.optim, cfg.name)
    elif hasattr(apex.optimizers, cfg.name):
        cls = getattr(apex.optimizers, cfg.name)
    else:
        raise ValueError(f'optimizer {optimizer} is invalid.')
    kwargs = {}
    for k, v in cfg.items():
        if k in inspect.signature(cls.__init__).parameters:
            kwargs[k] = v
    return cls(parameters, **kwargs)
