import os
import random
import numpy as np
import torch
import argparse
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from attrdict import AttrDict
from collections import defaultdict
from sklearn import preprocessing
import itertools

import src.utils.validation as validation
import src.utils.decomposition as decomposition
import src.models as models
import src.datasets as datasets

def get_optimizer(model, optimizer, lr=1e-3, weight_decay=1e-4):
    if optimizer is 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer is 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f'optimizer {optimizer} is invalid.')

def acronym(name):
    name = re.sub(r'(^[0-9a-zA-Z]{5,}(?=_))|((?<=_)[0-9a-zA-Z]*)',
                  lambda m: str(m.group(1) or '')[
                      :3] + str(m.group(2) or '')[:1],
                  name)
    name = name.replace('_', '.')
    return name

def plot_features_2d(xx, yy, labels, out):
    plt.figure(figsize=(8, 8))
    xx, yy, labels = validation.check_array(
        xx, yy, labels, check_size=True, dtype=[np.float, np.float, np.str])
    labels_unique = validation.check_array(labels, unique=True, sort=True, dtype=np.str)
    for i, label in enumerate(labels_unique):
        if label not in labels:
            continue
        idx = np.where(labels==label)
        x = xx[idx]
        y = yy[idx]
        plt.scatter(x, y, s=8.0, label=label)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
               borderaxespad=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_cm(cm, xlabels, ylabels, out):
    plt.figure(figsize=(8 * len(xlabels) / len(ylabels), 8))
    cmap = plt.get_cmap('Blues')
    cm_norm = preprocessing.normalize(cm, axis=0, norm='l1')
    plt.imshow(cm_norm.T, interpolation='nearest', cmap=cmap, origin='lower')
    ax = plt.gca()
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    plt.setp(ax.get_yticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")
    thresh = 1. / 1.75
    for i, j in itertools.product(range(len(xlabels)), range(len(ylabels))):
        num = "{}".format(cm[i, j])
        color = "white" if cm_norm[i, j] > thresh else "black"
        ax.text(i, j, num, fontsize=8, color=color, ha='center', va='center')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def logger(log, epoch, outdir):
    for k, v in log.items():
        plt.figure(figsize=(10, 6))
        out = f'{outdir}/{k}_{epoch}.png'
        yy = np.array(v)
        xx = np.array(range(len(v))) + 1
        plt.plot(xx, yy)
        plt.xlabel('epoch')
        plt.ylabel(k)
        plt.xlim((min(xx), max(xx)))
        plt.tight_layout()
        plt.savefig(out)
        plt.close()


def perturb(x, noise_rate=0.1):
    xt = x.clone()
    noise = torch.randn_like(x) * noise_rate
    xt += noise
    return xt


SEED_VALUE = 1234
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)

flags = AttrDict(
    # setup params
    batch_size=128,
    num_workers=4,
    num_epochs=100,
    use_perturb=False,
    save_step=10,
    save_path='/content/model.pt',
    # model params
    model='ResNet34',
    z_dim=512,
    # optimizer params
    optimizer='Adam',
    lr=1e-3,
    weight_decay=1e-4,
    # log params
    outdir='/content',
    eval_step=10,
)

parser = argparse.ArgumentParser()
parser.add_argument('path_to_hdf', type=validation.is_hdf,
                    help='path to hdf file.')
args = parser.parse_args()

hdf_file = args.path_to_hdf

train_set = datasets.HDF5Dataset(hdf_file)
_, test_set = train_set.split_dataset(0.7)
target_dict = {}
for _, target in train_set:
    target_dict[target['target_index']] = acronym(target['target_name'])
print('target_dict:', target_dict)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=flags.batch_size, num_workers=flags.num_workers,
    shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=flags.batch_size, num_workers=flags.num_workers,
    shuffle=False, drop_last=True)

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

model = models.VAE(flags.model, in_channels=4, z_dim=flags.z_dim).to(device)
optimizer = get_optimizer(model, flags.optimizer, lr=flags.lr, weight_decay=flags.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=2, T_mult=2)

log = defaultdict(lambda: [])
for epoch in range(1, flags.num_epochs):
    print(f'---------- training at epoch {epoch} ----------')
    model.train()
    loss = defaultdict(lambda: 0)
    for x, targets in tqdm(train_loader):
        x = x.to(device)
        x_generated, z, z_mean, z_var = model(x)
        loss_step = model.criterion(x, x_generated, z_mean, z_var)

        if flags.use_perturb:
            xt = perturb(x)
            xt_generated, zt, zt_mean, zt_var = model(x)
            loss_step += model.criterion(xt, xt_generated, zt_mean, zt_var)

        optimizer.zero_grad()
        loss_step.backward()
        optimizer.step()

        loss["train"] += loss_step.item()

    scheduler.step()
    log['train_loss'].append(loss['train'])

    print(f'train loss: {loss["train"]:.3f}')

    if epoch % flags.eval_step == 0:
        print(f'---------- evaluating at epoch {epoch} ----------')
        model.eval()
        result = defaultdict(lambda: [])
        with torch.no_grad():
            for x, targets in tqdm(test_loader):
                x = x.to(device)
                x_generated, z, z_mean, z_var = model(x)
                result['true'].append(targets['target_index'])
                result['z'].append(z.cpu())

        trues = torch.cat(result['true']).numpy()
        z = torch.cat(result['z']).numpy()
        z = decomposition.TSNE(n_components=2).fit_transform(z)
        logger(log, epoch, flags.outdir)
        z_labels = [f'{i}-{target_dict[i]}' for i in trues]
        plot_features_2d(z[:, 0], z[:, 1], z_labels, f'{flags.outdir}/z_{epoch}.png')

    if epoch % flags.save_step == 0:
        print(f'---------- saving check point at epoch {epoch} ----------')
        torch.save(model.state_dict(), flags.save_path)
