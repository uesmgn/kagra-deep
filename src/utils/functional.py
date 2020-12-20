import torch
import numpy as np
from collections import abc
import re
import warnings
import matplotlib.colors as mc
import colorsys
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage

__all__ = [
    "cosine_similarity",
    "normalize",
    "pca",
    "cmap_with_marker",
    "darken",
    "acronym",
    "to_device",
    "flatten",
    "tensordict",
    "compute_serial_matrix",
    "sample_from_each_class",
]


def sample_from_each_class(y, sample_size=10, random_seed=42, replace=False):
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced = {}
    for level, gb_idx in groupby_levels.items():
        indices = np.random.choice(gb_idx, size=sample_size, replace=replace).tolist()
        balanced[level] = indices
    return balanced


def seriation(Z, N, cur_index):
    """
    input:
        - Z is a hierarchical tree (dendrogram)
        - N is the number of points given to the clustering process
        - cur_index is the position in the tree for the recursive traversal
    output:
        - order implied by the hierarchical tree Z

    seriation computes the order implied by a hierarchical tree (dendrogram)
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(X, method="ward"):
    """
    input:
        - dist_mat is a distance matrix
        - method = ["ward","single","average","complete"]
    output:
        - seriated_dist is the input dist_mat,
          but with re-ordered rows and columns
          according to the seriation, i.e. the
          order implied by the hierarchical tree
        - res_order is the order implied by
          the hierarhical tree
        - res_linkage is the hierarhical tree (dendrogram)

    compute_serial_matrix transforms a distance matrix into
    a sorted distance matrix according to the order implied
    by the hierarchical tree (dendrogram)
    """
    dist_mat = squareform(pdist(X))
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


def cosine_similarity(x):
    x = x / x.norm(dim=-1)[:, None]
    return torch.mm(x, x.transpose(0, 1))


def normalize(x, axis=1):
    return (x - np.mean(x, axis=axis, keepdims=True)) / np.std(x, axis=axis, keepdims=True)


def pca(x, k, center=True):
    n = x.shape[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    h = torch.eye(n) - h
    h = h.to(x.device)
    x_center = torch.mm(h.double(), x.double())
    u, s, v = torch.svd(x_center)
    components = v[:k].t()
    return components


def cmap_with_marker(i, cmap="tab20b", markers="os^*p"):
    cmap = plt.cm.get_cmap(cmap)
    c = cmap.colors[i % cmap.N]
    m = markers[int(i / cmap.N)]
    return c, m


def darken(c, amount=0.5):
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def acronym(name):
    name = re.sub(
        r"(^[0-9a-zA-Z]{5,}(?=_))|((?<=_)[0-9a-zA-Z]*)",
        lambda m: str(m.group(1) or "")[:3] + str(m.group(2) or "")[:1],
        name,
    )
    name = name.replace("_", ".")
    return name


def getattr(d, name):
    assert isinstance(d, types.ModuleType)
    keys = []
    for key, obj in inspect.getmembers(d):
        if inspect.isclass(obj) and d.__name__ in obj.__module__:
            keys.append(key)
    for key in keys:
        if key.lower() == name.lower():
            return vars(d)[key]
    raise ValueError("Available class names are {}, but input is {}.".format(keys, name))


def to_device(device, *args):
    ret = []
    for arg in args:
        if torch.is_tensor(arg):
            ret.append(arg.to(device, non_blocking=True))
        elif isinstance(arg, abc.Sequence):
            ret.extend(to_device(device, *arg))
        else:
            raise ValueError(f"Input is invalid argument type: {type(arg)}.")
    return tuple(ret)


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class TensorDict(dict):
    def __init__(self, d=None):
        super().__init__()
        if isinstance(d, abc.MutableMapping):
            self.update(d)

    def cat(self, d, dim=0):
        if isinstance(d, abc.MutableMapping):
            for key, x in d.items():
                assert torch.is_tensor(x)
                x = x.detach().cpu()
                if key not in self:
                    self[key] = x
                else:
                    old = self[key]
                    self[key] = torch.cat([old, x], dim)
        else:
            raise ValueError("Invalid arguments.")

    def stack(self, d, dim=0):
        if isinstance(d, abc.MutableMapping):
            for key, x in d.items():
                assert torch.is_tensor(x)
                x = x.unsqueeze(dim).detach().cpu()
                if key not in self:
                    self[key] = x
                else:
                    old = self[key]
                    self[key] = torch.cat([old, x], dim)
        else:
            raise ValueError("Invalid arguments.")

    def mean(self, key, keep_dim=-1):
        x = self[key]
        if torch.is_tensor(x):
            if isinstance(keep_dim, int):
                dims = list(range(x.ndim))
                dims.pop(keep_dim)
                n = x.shape[keep_dim]
                x = x.permute(keep_dim, *dims).contiguous().view(n, -1).mean(-1)
            else:
                x = x.mean()
        self.update({key: x})
        return self

    def flatten(self, key):
        value = self[key]
        new = {}
        if torch.is_tensor(value):
            value = value.view(-1).tolist()
            if len(value) >= 10:
                warnings.warn("dimention is too large.")
                new[key] = value
            else:
                total = 0
                for i, x in enumerate(value):
                    new_key = "{}_{}".format(key, i)
                    new[new_key] = x
                    total += x
                new["{}_total".format(key)] = total
        else:
            new[key] = value

        self.pop(key)
        self.update(new)
        return self


def tensordict(d=None):
    return TensorDict(d)
