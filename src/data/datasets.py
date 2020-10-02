import os
import h5py
import copy
import torch
import numpy as np
from torch.utils import data
from collections import defaultdict
from skimage import color
import torchvision.transforms.functional as tf
import hydra

class HDF5Dataset(data.Dataset):
    def __init__(self, cfg, transform=None):
        super().__init__()
        self.cache = []
        self.root = hydra.utils.to_absolute_path(cfg.path)
        self.transform = transform

        self.targets = [target.lower() for target in cfg.targets]
        assert 'other' not in self.targets
        self.use_other = cfg.use_other
        if self.use_other:
            self.targets.append('other')
        self.shape = tuple(cfg.shape)
        self.channels = tuple(cfg.channels)
        with h5py.File(self.root, 'r') as fp:
            self._init_data_cache(fp)
        print(f'Successfully loaded {len(self.counter)} classes, {self.__len__()} items.')
        print(self.counter)

    @property
    def counter(self):
        cnt = defaultdict(lambda: 0)
        for _, it in self.cache:
            cnt[it] += 1
        cnt = sorted(cnt.items(), key=lambda x: x[0])
        return dict(cnt)

    def __getitem__(self, i):
        ref, it = self.cache[i]
        with h5py.File(self.root, 'r') as fp:
            item = fp[ref]
            x = self._load_data(item)
            x = x.index_select(0, torch.LongTensor(self.channels))
        if self.transform is not None:
            x = self.transform(x)
        if isinstance(x, tuple):
            return (*x, it)
        return (x, it)

    def _load_data(self, item):
        x = np.zeros(self.shape, dtype=np.uint8)
        item.read_direct(x)
        x = tf.to_tensor(x.transpose(1, 2, 0))
        return x

    def __len__(self):
        return len(self.cache)

    def _target_index(self, target):
        target = target.lower()
        if target not in self.targets:
            if self.use_other:
                target = 'other'
            else:
                return None
        return self.targets.index(target)

    def _init_data_cache(self, item):
        if hasattr(item, 'values'):
            # if item is group
            for v in item.values():
                self._init_data_cache(v)
        else:
            target = item.attrs['target']
            it = self._target_index(target)
            if it is not None:
                self.cache.append((item.ref, it))

    def get_label(self, i):
        _, it = self.cache[i]
        return it, self.targets[it]

    def split(self, alpha=0.8):
        N_train = int(self.__len__() * alpha)
        idx = np.arange(self.__len__())
        np.random.shuffle(idx)
        train_idx, test_idx = idx[:N_train], idx[N_train:]

        train_set = copy.copy(self)
        train_cache = [self.cache[i] for i in train_idx]
        train_set.cache = train_cache

        test_set = copy.copy(self)
        test_cache = [self.cache[i] for i in test_idx]
        test_set.cache = test_cache

        return train_set, test_set

    def copy(self):
        return copy.copy(self)

    def split_balancing(self, num_per_label=10):
        balenced_dict = defaultdict(lambda: [])
        idx = np.arange(self.__len__())
        for i in idx:
            _, it = self.cache[i]
            if len(balenced_dict[it]) < num_per_label:
                balenced_dict[it].append(i)
        uni_idx = np.array([i for v in balenced_dict.values() for i in v]).astype(np.integer)
        rem_idx = np.array(list(set(idx) - set(uni_idx))).astype(np.integer)

        uni_set = copy.copy(self)
        uni_cache = [self.cache[i] for i in uni_idx]
        uni_set.cache = uni_cache

        rem_set = copy.copy(self)
        rem_cache = [self.cache[i] for i in rem_idx]
        rem_set.cache = rem_cache

        return uni_set, rem_set

    def split_balancing_by_rate(self, rate_per_label=0.1):
        balenced_dict = defaultdict(lambda: [])
        idx = np.arange(self.__len__())
        num_per_labels = []
        for k, v in self.counter.items():
            num_per_labels.append((k, int(v * rate_per_label)))
        num_per_labels = dict(num_per_labels)
        for i in idx:
            _, it = self.cache[i]
            num_per_label = num_per_labels[it]
            if len(balenced_dict[it]) < num_per_label:
                balenced_dict[it].append(i)

        uni_idx = np.array([i for v in balenced_dict.values() for i in v]).astype(np.integer)
        rem_idx = np.array(list(set(idx) - set(uni_idx))).astype(np.integer)

        uni_set = copy.copy(self)
        uni_cache = [self.cache[i] for i in uni_idx]
        uni_set.cache = uni_cache

        rem_set = copy.copy(self)
        rem_cache = [self.cache[i] for i in rem_idx]
        rem_set.cache = rem_cache

        return uni_set, rem_set
