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
import src.models as models
import src.datasets as datasets


def acronym(name):
    name = re.sub(r'(^[0-9a-zA-Z]{5,}(?=_))|((?<=_)[0-9a-zA-Z]*)',
                  lambda m: str(m.group(1) or '')[
                      :3] + str(m.group(2) or '')[:1],
                  name)
    name = name.replace('_', '.')
    return name


def plot_cm(cm, xlabels, ylabels, out):
    plt.figure(figsize=(6 * len(xlabels) / len(ylabels), 6))
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
        ax.text(i, j, num, fontsize=6, color=color, ha='center', va='center')
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
    model='ResNet34',
    batch_size=64,
    num_workers=4,
    num_epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    num_classes=22,
    num_classes_over=100,
    outdir='/content',
    eval_step=10,
    num_heads=5,
    avg_for_heads=True
)

parser = argparse.ArgumentParser()
parser.add_argument('path_to_hdf', type=validation.is_hdf,
                    help='path to hdf file.')
args = parser.parse_args()

hdf_file = args.path_to_hdf

train_set, test_set = datasets.HDF5Dataset(hdf_file).split_dataset(0.7)
target_dict = {}
for _, target in train_set:
    target_dict[target['target_index']] = acronym(target['target_name'])
print('target_dict:', target_dict)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=flags.batch_size, num_workers=flags.num_workers,
    shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=flags.batch_size, num_workers=flags.num_workers,
    shuffle=False)

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

model = models.IIC(flags.model, in_channels=4, num_classes=flags.num_classes,
                   num_classes_over=flags.num_classes_over,
                   num_heads=flags.num_heads,
                   perturb_fn=lambda x: perturb(x)).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=flags.lr,
                            weight_decay=flags.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=2, T_mult=2)

log = defaultdict(lambda: [])
for epoch in range(1, flags.num_epochs):
    model.train()
    model.initialize_headers_weights()
    loss = defaultdict(lambda: 0)
    head_selecter = torch.zeros(flags.num_heads).to(device)
    for x, targets in tqdm(train_loader):
        x = x.to(device)
        y_outputs, y_over_outputs = model(x)
        yt_outputs, yt_over_outputs = model(x, perturb=True)
        loss_step_for_each_head = []
        for i in range(flags.num_heads):
            y, y_over = y_outputs[i], y_over_outputs[i]
            yt, yt_over = yt_outputs[i], yt_over_outputs[i]
            loss_step_head = model.criterion(y, yt) + model.criterion(y_over, yt_over)
            loss_step_for_each_head.append(loss_step_head)
        loss_step_for_each_head = torch.stack(loss_step_for_each_head)
        loss_step = torch.sum(loss_step_for_each_head)
        if flags.avg_for_heads:
            loss_step /= flags.num_heads
        head_selecter += loss_step_for_each_head

        optimizer.zero_grad()
        loss_step.backward()
        optimizer.step()

        loss['train'] += loss_step.item()

    scheduler.step()
    print(f'train loss at epoch {epoch} = {loss["train"]:.3f}')
    log['train_loss'].append(loss['train'])
    print('head_selecter:', head_selecter)
    best_head_idx = head_selecter.argmin(dim=-1).item()
    print('best_head_idx:', best_head_idx)

    if epoch % flags.eval_step != 0:
        continue

    model.eval()
    result = defaultdict(lambda: [])
    with torch.no_grad():
        for x, targets in tqdm(test_loader):
            x = x.to(device)
            y, y_over = model(x, head_index=best_head_idx)
            result['pred'].append(y.argmax(dim=-1).cpu())
            result['pred_over'].append(y_over.argmax(dim=-1).cpu())
            result['true'].append(targets['target_index'])
    preds = torch.cat(result['pred']).numpy()
    preds_over = torch.cat(result['pred_over']).numpy()
    trues = torch.cat(result['true']).numpy()
    cm = np.zeros((flags.num_classes, max(trues) + 1), dtype=np.int)
    for i, j in zip(preds, trues):
        cm[i, j] += 1
    cm_ylabels = [f'{i}-{target_dict[i]}' for i in range(max(trues)+1)]
    plot_cm(cm, list(range(flags.num_classes)), cm_ylabels,
            f'{flags.outdir}/cm_{epoch}.png')
    cm_over = np.zeros((flags.num_classes_over, max(trues) + 1), dtype=np.int)
    for i, j in zip(preds_over, trues):
        cm_over[i, j] += 1
    plot_cm(cm_over, list(range(flags.num_classes_over)), list(range(max(trues) + 1)),
            f'{flags.outdir}/cm_over_{epoch}.png')
    logger(log, epoch, flags.outdir)
