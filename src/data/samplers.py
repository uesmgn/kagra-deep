import torch
import torch.utils.data as data
import torchvision
from collections import defaultdict


class BalancingSampler(data.sampler.Sampler):

    def __init__(self, dataset, callback_get_label, num_samples=None):

        self.indices = list(range(len(dataset)))
        self.callback_get_label = callback_get_label
        self.num_samples = num_samples or len(self.indices)
        label_to_count = defaultdict(lambda: 0)
        labels = []
        for idx in self.indices:
            label, _ = callback_get_label(idx)
            label_to_count[label] += 1
            labels.append(label)
        weights = [1. / label_to_count[labels[idx]] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
