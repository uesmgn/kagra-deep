import os
import h5py
import copy
import torch
import numpy as np
from torch.utils import data
from skimage import color
import torchvision.transforms.functional as tf

class HDF5Dataset(data.Dataset):
    def __init__(self, path_to_hdf, transform=None):
        super().__init__()
        self.data_cache = []

        path_to_hdf = os.path.abspath(path_to_hdf)
        self.root = path_to_hdf
        self.transform = transform
        print('Appending data to cache...')
        with h5py.File(self.root, 'r') as fp:
            self._init_data_cache(fp)
        print(f'Successfully loaded {self.__len__()} items from {self.root}.')

    def __getitem__(self, i):
        ref = self.data_cache[i]
        with h5py.File(self.root, 'r') as fp:
            item = fp[ref]
            target = dict(item.attrs)
            img = np.array(item[:])
        img = tf.to_tensor(img.transpose(1, 2, 0))
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data_cache)

    def _init_data_cache(self, item):
        if hasattr(item, 'values'):
            # if item is group
            for it in item.values():
                self._init_data_cache(it)
        else:
            # if item is dataset
            self.data_cache.append(item.ref)

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
