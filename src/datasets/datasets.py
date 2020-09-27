import os
import h5py
import copy
import torch
import numpy as np
from torch.utils import data
from skimage import color
import torchvision.transforms.functional as tf

class HDF5Dataset(data.Dataset):
    def __init__(self, path_to_hdf, transform_fn=None, perturb_fn=None):
        super().__init__()
        self.data_cache = []

        path_to_hdf = os.path.abspath(path_to_hdf)
        self.root = path_to_hdf
        self.transform_fn = transform_fn
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
            img = np.array(item[:]).astype(np.uint8)
        img = tf.to_pil_image(img)
        if self.transform_fn is not None:
            img = self.transform_fn(img)
        if self.perturb_fn is not None:
            perturb_img = self.perturb_fn(img)
            return img, perturb_img, target
        else:
            return img, target

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
