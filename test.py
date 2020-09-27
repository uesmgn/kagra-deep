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
import src.datasets as datasets


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
    alpha=10,
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

argument_fn = torchvision.transforms.Compose([
    torchvision.transforms.ToPilImage(),
    torchvision.transforms.RandomCrop((224, 224 // 1.2)),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
])

perturb_fn = torchvision.transforms.Compose([
    torchvision.transforms.RandomChoice([
        torchvision.transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),
        torchvision.transforms.Lambda(lambda x: imp.gaussian_blur(x, 7, 10.)),
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
                               argument_fn=argument_fn,
                               n_arguments=10,
                               perturb_fn=perturb_fn)
labeled_set, unlabeled_set = dataset.balanced_dataset('target_index', num_per_label)
test_set, _ = dataset.balanced_dataset('target_index', 32)

print('len(dataset): ', len(dataset))
print('len(train_set): ', len(labeled_set) + len(unlabeled_set))
print('len(test_set): ', len(test_set))
print('len(labeled_set): ', len(labeled_set))
print('len(unlabeled_set): ', len(unlabeled_set))

path_to_model = args.path_to_model
outdir = args.path_to_outdir or flags.outdir
eval_step = args.eval_step or flags.eval_step
in_channels = len(flags.use_channels)
alpha = flags.alpha or len(labeled_set) * 0.1

labeled_loader = torch.utils.data.DataLoader(
    labeled_set, batch_size=flags.batch_size, num_workers=flags.num_workers,
    shuffle=True, drop_last=True)
unlabeled_loader = torch.utils.data.DataLoader(
    unlabeled_set, batch_size=flags.batch_size, num_workers=flags.num_workers,
    shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=flags.batch_size, num_workers=flags.num_workers,
    shuffle=False)

for x, xt, targets in tqdm(train_loader):
    print(x.shape)
    print(xt.shape)
    break
