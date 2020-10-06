import os
import h5py
import torch
from collections import abc, defaultdict
from torch.utils import data
import torchvision.transforms.functional as ttf
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import copy

__class__ = [
    'HDF5'
]

class HDF5(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, target_tag="target", shape=None):
        super().__init__()
        self.root = os.path.abspath(root)
        self.transform = transform
        self.__target_transform = target_transform
        self.__target_validate = lambda x: True
        self.target_tag = target_tag
        self.shape = shape

        self.fp = self.__fp()

        self.cache = None
        print("Initializing dataset cache...")
        try:
            _ = self.init_cache()
        except:
            raise RuntimeError(f"Failed to load items from {self.root}.")
        print(f"Successfully loaded {len(self)} items in cache from {self.root}.")

    @property
    def targets(self):
        tmp = []
        for _, target in self.cache:
            if self.__target_transform is not None:
                target = self.__target_transform(target)
            tmp.append(target)
        return tmp

    @property
    def counter(self):
        cnt = defaultdict(lambda: 0)
        for i in range(len(self)):
            _, target = self.cache[i]
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
        self.__target_validate = callback
        self.init_cache()

    def get_target(self, i):
        _, target = self.cache[i]
        if self.__target_transform is not None:
            target = self.__target_transform(target)
        return target

    def split(self, train_size=0.7, stratify=None):
        idx = list(range(len(self)))
        train_idx, test_idx = train_test_split(idx,
                                               train_size=train_size,
                                               random_state=123,
                                               stratify=stratify)
        train_set = copy.copy(self).init_cache(train_idx)
        test_idx = copy.copy(self).init_cache(test_idx)
        return train_set, test_idx

    def init_cache(self, indices=None):
        self.cache = []
        try:
            self.cache = self.__children(self.fp)
        except:
            self.fp = self.__fp()
            self.cache = self.__children(self.fp)
        if isinstance(indices, list):
            self.cache = [self.cache[i] for i in indices]
        return self

    def __children(self, d):
        items = []
        for k, v in d.items():
            if isinstance(v, abc.MutableMapping):
                items.extend(self.__children(v))
            else:
                target = v.attrs['target']
                val = self.__target_validate(target)
                if val is not None:
                    if not isinstance(val, (str, int)):
                        warnings.warn('target should be int or str.')
                    items.append((v.ref, target))
        return items

    def __fp(self):
        return h5py.File(self.root, 'r', libver='latest')

    def __getitem__(self, i):
        ref, target = self.cache[i]
        try:
            item = self.fp[ref]
        except:
            self.fp = self.__fp()
            item = self.fp[ref]
        x = self.__load_data(item)
        if self.transform is not None:
            x = self.transform(x)
        if self.__target_transform is not None:
            target = self.__target_transform(target)
        return x, target

    def __len__(self):
        return len(self.cache)

    def __load_data(self, item):
        if self.shape is not None:
            x = np.empty(self.shape, dtype=np.uint8)
            item.read_direct(x)
        else:
            x = np.array(item[:])
        x = ttf.to_tensor(x.transpose(1, 2, 0))
        return x
