import torch
import torch.utils.data as data
import torchvision
from collections import defaultdict

__class__ = ["Balancer"]


class Balancer(data.sampler.Sampler):
    def __init__(self, dataset, expansion=5.0):
        self.dataset = dataset
        if isinstance(expansion, float):
            self.num_samples = int(len(dataset) * self.expansion)
        elif isinstance(expansion, int):
            self.num_samples = expansion
        else:
            raise ValueError("Invalid argument.")
        self.init()

    def init(self):
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
