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

def _global(name):
    keys = [key for key in globals().keys() if key in __class__]
    for key in keys:
        if key.lower() == name.lower():
            return globals()[key]
    raise ValueError("Available class names are {}.".format(keys))

def get_dataset(name, **kwargs):
    dataset = _global(name)(**kwargs)
    return dataset

class HDF5(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, target_tag="target", shape=None):
        super().__init__()
        self.root = os.path.abspath(root)
        self.transform = transform
        self.__target_transform = target_transform
        self.__target_validate = lambda x: True
        self.target_tag = target_tag
        self.shape = shape

        self.fp = None
        self.open_once()

        self.cache = None
        print("Initializing dataset cache...")
        try:
            _ = self.init_cache()
        except:
            raise RuntimeError(f"Failed to load items from {self.root}.")
        print(f"Successfully loaded {len(self)} items in cache from {self.root}.")

    def open_once(self):
        self.fp = h5py.File(self.root, 'r', libver='latest', swmr=True)

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
            self.open_once()
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

    def __getitem__(self, i):
        ref, target = self.cache[i]
        try:
            item = self.fp[ref]
        except:
            self.open_once()
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

#
#     def split_balancing(self, num_per_label=10):
#         balenced_dict = defaultdict(lambda: [])
#         idx = np.arange(self.__len__())
#         for i in idx:
#             _, it = self.cache[i]
#             if len(balenced_dict[it]) < num_per_label:
#                 balenced_dict[it].append(i)
#         uni_idx = np.array([i for v in balenced_dict.values() for i in v]).astype(np.integer)
#         rem_idx = np.array(list(set(idx) - set(uni_idx))).astype(np.integer)
#
#         uni_set = copy.copy(self)
#         uni_cache = [self.cache[i] for i in uni_idx]
#         uni_set.cache = uni_cache
#
#         rem_set = copy.copy(self)
#         rem_cache = [self.cache[i] for i in rem_idx]
#         rem_set.cache = rem_cache
#
#         return uni_set, rem_set
#
#     def split_balancing_by_rate(self, rate_per_label=0.1):
#         balenced_dict = defaultdict(lambda: [])
#         idx = np.arange(self.__len__())
#         num_per_labels = []
#         for k, v in self.counter.items():
#             num_per_labels.append((k, int(v * rate_per_label)))
#         num_per_labels = dict(num_per_labels)
#         for i in idx:
#             _, it = self.cache[i]
#             num_per_label = num_per_labels[it]
#             if len(balenced_dict[it]) < num_per_label:
#                 balenced_dict[it].append(i)
#
#         uni_idx = np.array([i for v in balenced_dict.values() for i in v]).astype(np.integer)
#         rem_idx = np.array(list(set(idx) - set(uni_idx))).astype(np.integer)
#
#         uni_set = copy.copy(self)
#         uni_cache = [self.cache[i] for i in uni_idx]
#         uni_set.cache = uni_cache
#
#         rem_set = copy.copy(self)
#         rem_cache = [self.cache[i] for i in rem_idx]
#         rem_set.cache = rem_cache
#
#         return uni_set, rem_set
