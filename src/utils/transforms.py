import torch
import torchvision.transforms as tt
import torchvision.transforms.functional as ttf
import PIL
import random
import numbers
import types


def _get_image_size(x):
    if ttf._is_pil_image(x):
        return x.size
    elif isinstance(x, torch.Tensor) and x.ndim > 2:
        return x.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(x)))

def _to_pil_image(x):
    if ttf._is_pil_image(x):
        return x
    elif isinstance(x, torch.Tensor) and x.ndim in (1, 2, 3, 4):
        return ttf.to_pil_image(x)
    else:
        raise TypeError("Unexpected type {} or dimention".format(type(x)))

def _to_tensor(x):
    if ttf._is_pil_image(x):
        return ttf.to_tensor(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise TypeError("Unexpected type {}".format(type(x)))

def _lower(x):
    if isinstance(x, str):
        return x.lower()
    return x

class CenterMaximizedResizeCrop:
    def __init__(self, target_size):
        if isinstance(target_size, numbers.Number):
            self.target_size = (int(target_size), int(target_size))
        else:
            self.target_size = target_size

    def __call__(self, x):
        w, h = _get_image_size(x)
        tw, th = self.target_size
        x = _to_pil_image(x)
        if w >= tw and h >= th:
            ratio = max(tw, th) / min(w, h)
            x = ttf.resize(x, (int(h * ratio), int(w * ratio)))
        x = ttf.center_crop(x, (self.target_size))
        x = ttf.to_tensor(x)
        return x

class RandomMaximizedResizeCrop:
    def __init__(self, target_size):
        if isinstance(target_size, numbers.Number):
            self.target_size = (int(target_size), int(target_size))
        else:
            self.target_size = target_size

    def get_params(self, x, output_size):
        w, h = _get_image_size(x)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, x):
        w, h = _get_image_size(x)
        tw, th = self.target_size
        x = _to_pil_image(x)
        if w >= tw and h >= th:
            ratio = max(tw, th) / min(w, h)
            x = ttf.resize(x, (int(h * ratio), int(w * ratio)))
        i, j, h, w = self.get_params(x, self.target_size)
        x = ttf.crop(x, i, j, h, w)
        x = ttf.to_tensor(x)
        return x

class SelectIndices:
    def __init__(self, indices, dim=0):
        self.select_indices = tt.Lambda(
            lambda x: x.index_select(dim, torch.LongTensor(indices))
        )

    def __call__(self, x):
        x = _to_tensor(x)
        return self.select_indices(x)


class ToIndex:
    def __init__(self, targets, alt=None):
        assert isinstance(alt, (int, type(None)))
        targets_ = []
        for target in targets:
            if not isinstance(target, (int, str)):
                raise NotImplementedError('target must be str or int type.')
            target = _lower(target)
            targets_.append(target)
        self.to_index = lambda x: targets_.index(x) if x in targets_ else alt

    def __call__(self, x):
        x = _lower(x)
        return self.to_index(x)
