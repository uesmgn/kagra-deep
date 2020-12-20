import torch
import torch.utils.data as data
import torchvision
from collections import defaultdict
import numbers

__class__ = ["Balancer", "Upsampler"]


class Balancer(data.sampler.Sampler):
    def __init__(self, dataset, num_samples=10000, max_num_samples=100000):
        self.dataset = dataset
        if isinstance(num_samples, numbers.Number):
            self.num_samples = max(len(self.dataset), min(num_samples, max_num_samples))
        else:
            raise ValueError("Invalid argument.")
        self.indices = list(range(len(self.dataset)))
        targets = self.dataset.targets
        counter = defaultdict(lambda: 0)
        for idx in self.indices:
            label = targets[idx]
            counter[label] += 1
        weights = [1.0 / counter[targets[idx]] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class Upsampler(data.sampler.Sampler):
    def __init__(self, dataset, num_samples=10000, max_num_samples=1000000):
        self.dataset = dataset
        if isinstance(num_samples, numbers.Number):
            self.num_samples = max(len(self.dataset), min(num_samples, max_num_samples))
        else:
            raise ValueError("Invalid argument.")
        self.indices = list(range(len(self.dataset)))
        weights = torch.ones(len(self.indices)) / len(self.indices)
        self.weights = weights.double()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
