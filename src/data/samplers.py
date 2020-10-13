import torch
import torch.utils.data as data
import torchvision
from collections import defaultdict

__class__ = [
    'Balancer'
]

class Balancer(data.sampler.Sampler):

    def __init__(self, dataset, expansion=10.):
        self.__dataset = dataset
        self.expansion = expansion
        self.init()

    def init(self):
        self.indices = list(range(len(self.__dataset)))
        self.num_samples = int(len(self.__dataset) * self.expansion)
        targets = self.__dataset.targets
        counter = defaultdict(lambda: 0)
        for idx in self.indices:
            label = targets[idx]
            counter[label] += 1
        weights = [1. / counter[targets[idx]] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    @property
    def dataset(self):
        pass

    @dataset.setter
    def dataset(self, dataset):
        self.__dataset = dataset
        self.init()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
