import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import argparse
import re
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from attrdict import AttrDict
from collections import defaultdict
from sklearn import preprocessing
from itertools import cycle, product

from src.utils import validation
from src.utils import image_processing as imp
import src.models as models
from src.data import datasets, samplers


SEED_VALUE = 1234
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)

flags = AttrDict(
    # setup params
    batch_size=96,
    num_workers=4,
    num_epochs=5000,
    reinitialize_headers_weights=True,
    use_channels=[2],
    num_per_label=32,
    weights=(1., 10., 1.),
    input_shape=(479, 569),
    num_dataset_argmented=10000,
    # model params
    model='ResNet34',
    num_classes=22,
    num_classes_over=220,
    num_heads=5,
    # optimizer params
    optimizer='Adam',
    lr=1e-4,
    weight_decay=1e-4,
    # log params
    outdir='/content/run_iic_ssl',
    eval_step=10,
    avg_for_heads=True,
)

transform_fn = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: torch.stack([x[i] for i in flags.use_channels])),
])

argmentation_fn = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: torch.stack([x[i] for i in flags.use_channels])),

])

perturb_fn = torchvision.transforms.Compose([
    torchvision.transforms.RandomChoice([
        torchvision.transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),
        torchvision.transforms.Lambda(lambda x: imp.gaussian_blur(x, 5, 10.)),
    ]),
])

parser = argparse.ArgumentParser()
parser.add_argument('path_to_hdf', type=validation.is_hdf,
                    help='path to hdf file.')
parser.add_argument('-m', '--path_to_model', type=validation.is_file,
                    help='path to pre-trained model.state_dict().')
parser.add_argument('-o', '--path_to_outdir', type=validation.is_dir,
                    help='path to output directory.')
parser.add_argument('-e', '--eval_step', type=int,
                    help='evaluating step.')
parser.add_argument('-n', '--num_per_label', type=int,
                    help='num of labeled samples per label.')
args = parser.parse_args()

num_per_label = args.num_per_label or flags.num_per_label

dataset = datasets.HDF5Dataset(args.path_to_hdf,
                transform_fn=transform_fn,
                perturb_fn=perturb_fn)
# random num_per_label samples of each label are labeled.
labeled_set, _ = dataset.split_balanced('target_index', num_per_label)
labeled_loader = torch.utils.data.DataLoader(
    labeled_set,
    batch_size=flags.batch_size,
    num_workers=flags.num_workers,
    drop_last=True)
# all samples are unlabeled. A balanced sample is applied to these samples.
unlabeled_set = dataset.copy()
unlabeled_set.transform_fn = argmentation_fn
balanced_sampler = samplers.BalancedDatasetSampler(dataset,
    dataset.get_label, num_samples=flags.num_dataset_argmented)
unlabeled_loader = torch.utils.data.DataLoader(
    unlabeled_set,
    batch_size=flags.batch_size,
    num_workers=flags.num_workers,
    sampler=balanced_sampler,
    drop_last=True)
# 30% of all samples are test data.
test_set, _ = dataset.split(0.3)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=flags.batch_size,
    num_workers=flags.num_workers,
    shuffle=False)

print('len(dataset): ', len(dataset))
print('len(labeled_set): ', len(labeled_set))
print('len(unlabeled_set): ', len(unlabeled_set))
print('len(test_set): ', len(test_set))

targets = defaultdict(lambda:0)
for x, xt, target in tqdm(unlabeled_loader):
    for i in target['target_index']:
        targets[i.item()] += 1
for k, v in targets.items():
    print(k, v)

for i in range(5):
    plt.imshow(x[i][0], cmap='gray')
    plt.show()
    plt.imshow(xt[i][0], cmap='gray')
    plt.show()
