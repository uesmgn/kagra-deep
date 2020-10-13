import torch
import torch.utils.data as data
import torchvision
from collections import defaultdict
import numbers

__class__ = ["Balancer"]


class Balancer(data.sampler.Sampler):
    def __init__(self, dataset, expansion=1.0, max_num_samples=100000):
        self.dataset = dataset
        if isinstance(expansion, numbers.Integral):
            self.num_samples = expansion
        elif isinstance(expansion, numbers.Number):
            self.num_samples = int(len(dataset) * expansion)
        else:
            raise ValueError("Invalid argument.")
        self.num_samples = max(len(self.dataset), min(self.num_samples, max_num_samples))
        self.indices = list(range(len(self.dataset)))
        targets = self.dataset.targets
        counter = defaultdict(lambda: 0)
        for idx in self.indices:
            label = targets[idx]
            counter[label] += 1
        weights = [1.0 / counter[targets[idx]] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples
