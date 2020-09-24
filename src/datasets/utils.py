import h5py
import os

def get_hdf_items(path):
    cache = []
    def init_cache(item):
        if hasattr(item, 'values'):
            for it in item.values():
                init_cache(it)
        else:
            cache.append(item.ref)
    with h5py.File(path, 'r') as fp:
        init_cache(fp)
    return cache
