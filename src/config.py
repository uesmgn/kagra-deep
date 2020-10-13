import inspect
import types
import math
import logging
from collections import abc
from itertools import cycle
import torch
import numpy as np

try:
    import apex

    use_apex = True
except ImportError:
    use_apex = False


__types__ = ["basic", "co", "ss", "co+ss"]


def check_cfg(cfg):
    if "name" in cfg and isinstance(cfg.name, str):
        if isinstance(cfg.params, abc.MutableMapping):
            return cfg.name, dict(cfg.params)
        else:
            return cfg.name, {}
    raise NotImplementedError("cfg must include 'name' and 'params' (optional)")


def check_params(cfg):
    if isinstance(cfg, abc.MutableMapping):
        return dict(cfg)
    else:
        return {}


def instantiate(d, name):
    assert isinstance(d, types.ModuleType)
    keys = []
    for key, obj in inspect.getmembers(d):
        if inspect.isclass(obj) and d.__name__ in obj.__module__:
            keys.append(key)
    for key in keys:
        if key.lower() == name.lower():
            return vars(d)[key]
    raise ValueError("Available class names are {}, but input is {}.".format(keys, name))


def zip_loader(x, y, *args):
    args = [x, y, *args]
    max_idx = np.argmax([len(arg) for arg in args])
    max_arg = args.pop(max_idx)
    args = [cycle(arg) for arg in args]
    args.insert(max_idx, max_arg)
    return zip(*args)


class Config(object):
    def __init__(
        self,
        type="basic",
        device="cpu",
        transform=None,
        augment_transform=None,
        target_transform=None,
        sampler_callback=None,
        train_size=0.7,
        labeled_size=0.1,
    ):
        super().__init__()

        if type == "basic":
            logging.info("This is 'basic' configulation for basic learning.")
        elif type == "co":
            logging.info("This is 'co' configulation for co-training.")
        elif type == "ss":
            logging.info("This is 'ss' configulation for semi-supervised learning.")
        elif type == "co+ss":
            logging.info(
                "This is 'co+ss' configulation for co-training + semi-supervised learning."
            )
        else:
            raise NotImplementedError(f"configulation type must choose from {__types__}")

        self.type = type
        self.device = device
        self.transform = transform
        self.augment_transform = augment_transform
        self.target_transform = target_transform
        self.sampler_callback = sampler_callback
        self.train_size = train_size
        self.labeled_size = labeled_size

    def get_net(self, cfg, **kwargs):

        from .nn import nets

        name, params = check_cfg(cfg)
        params.update(kwargs)
        return instantiate(nets, name)(**params)

    def get_model(self, cfg, **kwargs):
        from .nn import models

        name, params = check_cfg(cfg)
        params.update(kwargs)
        return instantiate(models, name)(**params)

    def get_optim(self, cfg, **kwargs):

        name, params = check_cfg(cfg)
        params.update(kwargs)
        try:
            return instantiate(torch.optim, name)(**params)
        except ValueError as err:
            if not use_apex:
                raise err
        try:
            return instantiate(apex.optimizers, name)(**params)
        except ValueError as err:
            raise err

    def get_datasets(self, cfg, **kwargs):
        from .data import datasets

        name, params = check_cfg(cfg)
        params.update(kwargs)
        dataset = instantiate(datasets, name)(
            transform=self.transform, target_transform=self.target_transform, **params
        )
        train_set, test_set = dataset.split(self.train_size, stratify=dataset.targets)

        if self.type == "ss":
            l, u = train_set.split(self.labeled_size, stratify=train_set.targets)
            train_set = (l, u)
        elif self.type == "co":
            train_set = datasets.Co(train_set, self.augment_transform)
        elif self.type == "co+ss":
            l, u = train_set.split(self.labeled_size, stratify=train_set.targets)
            l, u = map(lambda x: datasets.Co(x, self.augment_transform), (l, u))
            train_set = (l, u)
        return train_set, test_set

    def get_loader(self, cfg, datasets):
        from torch.utils.data import DataLoader

        params = check_params(cfg)
        # if datasets is sequence, return zip of loaders
        if isinstance(datasets, abc.Sequence):
            loaders = (
                DataLoader(ds, sampler=self.sampler_callback(ds), **params) for ds in datasets
            )
            return zip_loader(*loaders)
        else:
            return DataLoader(datasets, sampler=self.sampler_callback(datasets), **params)
