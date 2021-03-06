import os
import h5py
import torch
from collections import abc, defaultdict
from itertools import cycle
from torch.utils import data
import torchvision.transforms as tf
import torchvision.transforms.functional as ttf
import numpy as np
from sklearn.model_selection import train_test_split
import inspect
import copy
import warnings

__class__ = ["HDF5", "co"]


class HDF5(data.Dataset):
    def __init__(self, root, transform, target_transform, target_tag="target", shape=None):
        super().__init__()
        self.__root = os.path.abspath(root)
        if not os.path.isfile(self.__root):
            raise FileNotFoundError(f"No such file or directory: '{self.__root}'")

        self.transform = transform
        self.__target_transform = target_transform
        self.__target_tag = target_tag
        self.__shape = shape

        self.__cache = None
        try:
            _ = self.__init_cache()
        except Exception as err:
            raise err

    @property
    def targets(self):
        tmp = []
        for _, target in self.__cache:
            if self.__target_transform is not None:
                target = self.__target_transform(target)
            tmp.append(target)
        return tmp

    @property
    def counter(self):
        cnt = defaultdict(lambda: 0)
        for i in range(len(self)):
            _, target = self.__cache[i]
            if self.__target_transform is not None:
                target = self.__target_transform(target)
            cnt[target] += 1
        cnt = sorted(cnt.items(), key=lambda x: x[1])
        return dict(cnt)

    @property
    def target_transform(self):
        pass

    @target_transform.setter
    def target_transform(self, callback):
        self.__target_transform = callback
        self.__init_cache()

    def split(self, train_size=0.7, stratify=None):
        idx = list(range(len(self)))
        train_idx, test_idx = train_test_split(idx, train_size=train_size, random_state=123, stratify=stratify)
        train_set = copy.copy(self).__select_indices(train_idx)
        test_set = copy.copy(self).__select_indices(test_idx)
        return train_set, test_set

    def sample(self, num_samples=1000, stratify=None):
        idx = list(range(len(self)))
        sample_idx, _ = train_test_split(idx, train_size=num_samples, random_state=123, stratify=stratify)
        return copy.copy(self).__select_indices(sample_idx)

    def __getitem__(self, i):
        ref, target = self.__cache[i]
        with self.__open(self.__root) as fp:
            item = fp[ref]
            x = self.__load(item)
        if self.transform is not None:
            x = self.transform(x)
        if self.__target_transform is not None:
            target = self.__target_transform(target)
        return x, target

    def __len__(self):
        return len(self.__cache)

    def __init_cache(self):
        self.__cache = []
        with self.__open(self.__root) as fp:
            self.__cache = self.__children(fp)
        return self

    def __select_indices(self, indices=None):
        if isinstance(indices, abc.Sequence):
            self.__cache = [self.__cache[i] for i in indices]
        return self

    def __children(self, d):
        items = []
        for k, v in d.items():
            if isinstance(v, abc.MutableMapping):
                items.extend(self.__children(v))
            else:
                target = v.attrs.get(self.__target_tag)
                if target is not None:
                    tmp = target
                    if self.__target_transform is not None:
                        tmp = self.__target_transform(tmp)
                    if tmp is None:
                        continue
                    if not isinstance(tmp, (str, int)):
                        warnings.warn("target data type must be int or str.")
                    items.append((v.ref, target))
        return items

    def __open(self, path):
        return h5py.File(path, "r", libver="latest")

    def __load(self, item):
        if self.__shape is not None:
            x = np.empty(self.__shape, dtype=np.uint8)
            item.read_direct(x)
        else:
            x = np.array(item[:])
        x = ttf.to_tensor(x.transpose(1, 2, 0))
        return x


def co(ds, tansform, augment, l=5):
    assert hasattr(ds, "transform")
    ds.transform = tf.Lambda(lambda x: (tansform(x), *[augment(x) for _ in range(l)]))
    return ds
