import os
import h5py
import copy
import torch
import numpy as np
from torch.utils import data
from collections import defaultdict
from skimage import color
import torchvision.transforms.functional as tf

class HDF5Dataset(data.Dataset):
    def __init__(self, path_to_hdf, transform_fn=None,
                 argument_fn=None, n_arguments=5, perturb_fn=None):
        super().__init__()
        self.data_cache = []

        self.root = os.path.abspath(path_to_hdf)
        self.transform_fn = transform_fn
        self.argument_fn = argument_fn
        self.n_arguments = n_arguments
        self.perturb_fn = perturb_fn
        print('Appending data to cache...')
        with h5py.File(self.root, 'r') as fp:
            self._init_data_cache(fp)
        print(f'Successfully loaded {self.__len__()} items from {self.root}.')

    def __getitem__(self, i):
        ref = self.data_cache[i]
        with h5py.File(self.root, 'r') as fp:
            item = fp[ref]
            target = dict(item.attrs)
            data = self._load_data(item)
        if self.transform_fn is not None:
            data = self.transform_fn(data)
        if self.argument_fn is not None:
            tmp = []
            for _ in range(self.n_arguments):
                x = self.argument_fn(data)
                tmp.append(x)
            data = torch.stack(tmp)
        if self.perturb_fn is not None:
            if data.ndim == 4:
                tmp = []
                for x in data:
                    x_ = self.perturb_fn(x)
                    tmp.append(x_)
                data_ = torch.stack(tmp)
            elif data.ndim == 3:
                data_ = self.perturb_fn(data)
            else:
                raise ValueError('ndim of data must be 3 or 4.')
            return data, data_, target
        return data, target

    def _load_data(self, item):
        img = np.array(item[:]).astype(np.uint8)
        img = tf.to_pil_image(img)
        return img

    def __len__(self):
        return len(self.data_cache)

    def _init_data_cache(self, item):
        if hasattr(item, 'values'):
            # if item is group
            for it in item.values():
                if item is not None:
                    self._init_data_cache(it)
                else:
                    print('item is NoneType object.')
        else:
            # if item is dataset
            if hasattr(item, 'ref'):
                self.data_cache.append(item.ref)
            else:
                print('item has no attributes "ref".')


    def split_dataset(self, alpha=0.8):
        N_train = int(self.__len__() * alpha)
        idx = np.arange(self.__len__())
        np.random.shuffle(idx)
        train_idx, test_idx = idx[:N_train], idx[N_train:]

        train_set = copy.copy(self)
        train_ref = [self.data_cache[i] for i in train_idx]
        train_set.data_cache = train_ref

        test_set = copy.copy(self)
        test_ref = [self.data_cache[i] for i in test_idx]
        test_set.data_cache = test_ref

        return train_set, test_set

    def balanced_dataset(self, attr, num_per_label=50):
        balenced_dict = defaultdict(lambda: [])
        idx = np.arange(self.__len__())
        with h5py.File(self.root, 'r') as fp:
            for i in idx:
                ref = self.data_cache[i]
                item = fp[ref]
                target = item.attrs[attr]
                if len(balenced_dict[target]) < num_per_label:
                    balenced_dict[target].append(i)
        uni_idx = np.array([i for v in balenced_dict.values() for i in v]).astype(np.integer)
        rem_idx = np.array(list(set(idx) - set(uni_idx))).astype(np.integer)

        uni_set = copy.copy(self)
        uni_ref = [self.data_cache[i] for i in uni_idx]
        uni_set.data_cache = uni_ref

        rem_set = copy.copy(self)
        rem_ref = [self.data_cache[i] for i in rem_idx]
        rem_set.data_cache = rem_ref

        return uni_set, rem_set
