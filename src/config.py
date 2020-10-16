import inspect
import types
import math
import logging
from collections import abc
from itertools import cycle
import torch
import numpy as np
from torch.utils.data import DataLoader

try:
    import apex

    use_apex = True
except ImportError:
    use_apex = False


__types__ = ["basic", "co", "ss", "co+ss"]

logger = logging.getLogger(__name__)


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


class ZipLoader(object):
    def __init__(self, *datasets, sampler_callback=None, **params):
        batch_size = params.pop("batch_size", 1)
        datasets = [x for x in datasets]
        max_idx = np.argmax([len(ds) for ds in datasets])
        max_ds = datasets.pop(max_idx)
        loaders = []
        for i, ds in enumerate(datasets):
            new_batch = len(ds) * batch_size // len(max_ds)
            loader = self.to_loader(ds, sampler_callback, batch_size=new_batch, **params)
            loaders.append(loader)
        loaders.insert(
            max_idx, self.to_loader(max_ds, sampler_callback, batch_size=batch_size, **params)
        )
        self.loaders = loaders
        self.len = len(loaders[max_idx])
        self.iterators = None

    def to_loader(self, ds, sampler_callback=None, **params):
        sampler = None
        if callable(sampler_callback):
            sampler = sampler_callback(ds)
        return DataLoader(ds, sampler=sampler, **params)

    def __iter__(self):
        self.iterators = [iter(loader) for loader in self.loaders]
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        sentinel = object()
        ret = []
        for it in self.iterators:
            el = next(it, sentinel)
            if el is sentinel:
                raise StopIteration()
            ret.append(el)
        return tuple(ret)


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
        logger.info(f"Successfully loaded {len(dataset)} data.", dataset.counter)
        train_set, test_set = dataset.split(self.train_size, stratify=dataset.targets)
        logger.info(f"train_set: {len(train_set)}\n", train_set.counter)
        logger.info(f"test_set: {len(test_set)}\n", test_set.counter)

        if self.type == "basic":
            if callable(self.sampler_callback):
                train_set.transform = self.augment_transform
        elif self.type == "ss":
            if callable(self.sampler_callback):
                train_set.transform = self.augment_transform
            l, u = train_set.split(self.labeled_size, stratify=train_set.targets)
            logger.info(f"labeled_set: {len(l)}\n", l.counter)
            logger.info(f"unlabeled_set: {len(u)}\n", u.counter)
            train_set = (l, u)
        elif self.type == "co":
            train_set = datasets.Co(train_set, self.augment_transform)
        elif self.type == "co+ss":
            l, u = train_set.split(self.labeled_size, stratify=train_set.targets)
            logger.info(f"labeled_set: {len(l)}\n", l.counter)
            logger.info(f"unlabeled_set: {len(u)}\n", u.counter)
            l, u = map(lambda x: datasets.Co(x, self.augment_transform), (l, u))
            train_set = (l, u)
        return train_set, test_set

    def get_loader(self, cfg, datasets, train=True):

        params = check_params(cfg)
        if not train:
            return DataLoader(datasets, **params)
        # if datasets is sequence, return zip of loaders
        if isinstance(datasets, abc.Sequence):
            return ZipLoader(*datasets, sampler_callback=self.sampler_callback, **params)
        else:
            return DataLoader(datasets, sampler=self.sampler_callback(datasets), **params)
