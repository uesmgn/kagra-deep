import os
import random
import numpy as np
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from attrdict import AttrDict
from collections import defaultdict

import src.utils.validation as validation
import src.models as models
import src.datasets as datasets

SEED_VALUE = 1234
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)

flags = AttrDict(
    batch_size=8,
    num_workers=1,
    num_epochs=100,
    lr=3e-4,
    weight_decay=1e-4,
)

parser = argparse.ArgumentParser()
parser.add_argument('path_to_hdf', type=validation.is_hdf,
                    help='path to hdf file.')
args = parser.parse_args()

hdf_file = args.path_to_hdf

train_set, test_set = datasets.HDF5Dataset(hdf_file).split_dataset(0.8)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=flags.batch_size, num_workers=flags.num_workers,
    shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=flags.batch_size, num_workers=flags.num_workers,
    shuffle=False)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
model = models.IIC('VGG11', in_channels=4, num_classes=10, num_classes_over=100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=flags.lr, weight_decay=flags.weight_decay)


for epoch in range(1, flags.num_epochs):
    model.train()
    loss = defaultdict(lambda: 0)
    for x, targets in tqdm(train_loader):
        x = x.to(device)
        y, y_over = model(x)
        yt, yt_over = model(x, perturb=True)
        loss_step = model.criterion(y, yt) + model.criterion(y_over, yt_over)
        optimizer.zero_grad()
        loss_step.backward()
        optimizer.step()
        loss['train'] += loss_step.item()
    print(f'train loss at epoch {epoch} = {loss["train"]:.3f}')
