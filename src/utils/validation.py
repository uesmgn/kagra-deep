import os
import re
import numpy as np

__all__ = [
    'is_dir', 'is_hdf', 'new_hdf', 'check_array'
]

def yn_input(text):
    choice = input(text).lower()
    while True:
        if choice in ['y', 'ye', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            choice = input(text + "enter [y/n] ").lower()


def is_dir(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return path
    else:
        raise None

def is_hdf(path):
    path = os.path.abspath(path)
    assert os.path.splitext(path)[-1] in ('.h5', '.hdf5')
    if os.path.exists(path):
        return path
    else:
        raise None

def new_hdf(path):
    path = os.path.abspath(path)
    assert os.path.splitext(path)[-1] in ('.h5', '.hdf5')
    if os.path.exists(path):
        if yn_input(f"{path} is already exists, overwrite? "):
            return path
        else:
            print("exit...")
            exit()
    else:
        return path

def check_array(*args, allow_2d=True, sort=False, reverse=False,
                unique=False, check_size=False, check_shape=False, dtype=None):
    if len(args) > 1 and check_size:
        if len(set([len(arr) for arr in args])) > 1:
            raise ValueError('Input arrays must have same length')
    if len(args) > 1 and check_shape:
        if len(set([arr.shape for arr in args])) > 1:
            raise ValueError('Input arrays must have same shape')
    values = []
    for i, arg in enumerate(args):
        if not hasattr(arg, '__len__') or isinstance(arg, str):
            raise ValueError('Input type must be array-like')
        if len(arg) < 1:
            raise ValueError('Input array must have elements')
        arr = _cast_array(arg)
        if arr.ndim is 1:
            # 1-D array
            # unique and sort operation only apply to 1-D array
            if unique:
                arr, index = np.unique(arr, return_index=True)
                arr = arr[index.argsort()]
            if sort:
                arr = np.sort(arr)
                if reverse:
                    arr = arr[::-1]
        elif allow_2d and arr.ndim is 2:
            pass
        else:
            raise ValueError('Input type must be 1-D array')
        if dtype is not None:
            arr = arr.astype(_list_get(dtype, i) or dtype)
        values.append(arr)
    return tuple(values) if len(values) > 1 else values[0]

def _list_get(arr, idx):
  try:
    return arr[idx]
  except:
    return None

def _cast_array(arg):
    arr = np.array(arg).astype(np.str)
    flat = np.ravel(arr)
    cast_type = None
    if all(list(map(lambda x: x.isnumeric(), flat))):
        cast_type = np.int
    else:
        try:
            _ = list(map(lambda x: float(x), flat))
            cast_type = np.float
        except:
            pass
    return arr.astype(cast_type) if cast_type else arr
