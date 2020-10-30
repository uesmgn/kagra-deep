from collections import abc
import numpy as np
from torch.utils.data import DataLoader
from . import datasets


class ZipLoader(object):
    def __init__(self, *datasets, sampler_callback=None, **kwargs):
        batch_size = kwargs.pop("batch_size", 1)
        datasets = [x for x in datasets]
        max_idx = np.argmax([len(ds) for ds in datasets])
        max_ds = datasets.pop(max_idx)
        loaders = []
        for i, ds in enumerate(datasets):
            new_batch = len(ds) * batch_size // len(max_ds)
            loader = self.to_loader(ds, sampler_callback, batch_size=new_batch, **kwargs)
            loaders.append(loader)
        loaders.insert(
            max_idx, self.to_loader(max_ds, sampler_callback, batch_size=batch_size, **kwargs)
        )
        self.loaders = loaders
        self.len = len(loaders[max_idx])
        self.iterators = None

    def to_loader(self, ds, sampler_callback=None, **kwargs):
        sampler = None
        if callable(sampler_callback):
            sampler = sampler_callback(ds)
        return DataLoader(ds, sampler=sampler, **kwargs)

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


class Dataset(object):
    def __init__(
        self,
        root,
        mode="basic",
        transform=None,
        target_transform=None,
        augment_transform=None,
        sampler_callback=None,
        target_tag="target",
        data_shape=None,
        train_size=0.8,
        labeled_size=0.8,
    ):
        self.dataset = datasets.HDF5(root, transform, target_transform, target_tag, data_shape)
        self.transform = transform
        self.target_transform = target_transform
        self.augment_transform = augment_transform
        self.sampler_callback = sampler_callback if callable(sampler_callback) else None
        self.__setup_datasets(mode, train_size, labeled_size)

    def __setup_datasets(self, mode="basic", train_size=0.8, labeled_size=0.8):
        train_set, test_set = self.dataset.split(train_size, stratify=self.dataset.targets)

        if mode == "basic":
            if self.sampler_callback is not None:
                train_set.transform = self.augment_transform

        elif mode == "ss":
            if self.sampler_callback is not None:
                train_set.transform = self.augment_transform
            labeled, unlabeled = train_set.split(labeled_size, stratify=train_set.targets)
            train_set = (labeled, unlabeled)

        elif mode == "co":
            train_set = datasets.Co(train_set, self.augment_transform)

        elif mode == "co+ss":
            labeled, unlabeled = train_set.split(labeled_size, stratify=train_set.targets)
            labeled, unlabeled = map(
                lambda x: datasets.Co(x, self.augment_transform), (labeled, unlabeled)
            )
            train_set = (labeled, unlabeled)

        self.train_set = train_set
        self.test_set = test_set

    def get_loader(self, train=True, **kwargs):

        if not train:
            return DataLoader(self.test_set, drop_last=False, **kwargs)
        # if datasets is sequence, return zip of loaders
        if isinstance(self.train_set, abc.Sequence):
            return ZipLoader(
                *self.train_set, sampler_callback=self.sampler_callback, drop_last=True, **kwargs
            )
        else:
            sampler = (
                self.sampler_callback(self.train_set) if callable(self.sampler_callback) else None
            )
            return DataLoader(self.train_set, sampler=sampler, drop_last=True, **kwargs)
