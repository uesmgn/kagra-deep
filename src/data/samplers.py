import torch
import torch.utils.data as data
import torchvision
from collections import defaultdict

__class__ = [
    'Balancer'
]

def _global(name):
    keys = [key for key in globals().keys() if key in __class__]
    for key in keys:
        if key.lower() == name.lower():
            return globals()[key]
    raise ValueError("Available class names are {}.".format(keys))

def get_sampler(name, dataset, **kwargs):
    sampler = _global(name)(dataset, **kwargs)
    return sampler

class Balancer(data.sampler.Sampler):

    def __init__(self, dataset, num_samples=None):
        assert hasattr(dataset, "targets")
        self.indices = list(range(len(dataset)))
        self.num_samples = num_samples or len(self.indices)
        targets = dataset.targets
        counter = defaultdict(lambda: 0)
        for idx in self.indices:
            label = targets[idx]
            counter[label] += 1
        weights = [1. / counter[targets[idx]] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
