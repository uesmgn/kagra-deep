import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import re
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from attrdict import AttrDict
from collections import defaultdict, Counter
from sklearn import preprocessing
from itertools import cycle, product
import apex
from apex import amp, optimizers

from src.utils import validation
from src.utils import image_processing as imp
import src.models as models
from src.data import datasets, samplers

def get_optimizer(model, optimizer, lr=1e-3, weight_decay=1e-4):
    if optimizer is 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer is 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer is 'FusedAdam':
        return apex.optimizers.FusedAdam(model.parameters(), lr=lr)
    else:
        raise ValueError(f'optimizer {optimizer} is invalid.')

def acronym(name):
    name = re.sub(r'(^[0-9a-zA-Z]{5,}(?=_))|((?<=_)[0-9a-zA-Z]*)',
                  lambda m: str(m.group(1) or '')[
                      :3] + str(m.group(2) or '')[:1],
                  name)
    name = name.replace('_', '.')
    return name


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
    for i, j in product(range(len(xlabels)), range(len(ylabels))):
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


SEED_VALUE = 1234
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)

NUM_BATCH_LABELED_DATA=32
NUM_LABELED_DATA=100
NUM_BATCH_UNLABELED_DATA=128
NUM_UNLABELED_DATA=10000
N_STEP=NUM_UNLABELED_DATA // NUM_BATCH_UNLABELED_DATA

flags = AttrDict(
    # setup params
    num_workers=4,
    num_epochs=5000,
    reinitialize_headers_weights=True,
    use_channels=[2],
    num_per_label=32,
    weights=(1., 10., 1.),
    labeled_batch_size=NUM_BATCH_LABELED_DATA,
    num_dataset_labeled=NUM_LABELED_DATA,
    unlabeled_batch_size=NUM_BATCH_UNLABELED_DATA,
    num_dataset_unlabeled=NUM_UNLABELED_DATA,
    test_batch_size=64,
    opt_level='O1',
    input_shape=(4, 479, 569),
    # model params
    model='ResNet34',
    num_classes=22,
    num_classes_over=50,
    num_heads=5,
    # optimizer params
    optimizer='FusedAdam',
    lr=1e-4,
    weight_decay=1e-4,
    # log params
    outdir='/content/run_iic_ssl',
    eval_step=10,
    avg_for_heads=True,
)

transform_fn = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop((479, 479)),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: torch.stack([x[i] for i in flags.use_channels]))
])

argmentation_fn = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((479, 479)),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: torch.stack([x[i] for i in flags.use_channels])),
])

perturb_fn = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: (transform_fn(x), argmentation_fn(x)))
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
                data_shape=flags.input_shape,
                transform_fn=perturb_fn)
# random num_per_label samples of each label are labeled.
labeled_set, _ = dataset.split_balanced('target_index', num_per_label)
labeled_loader = torch.utils.data.DataLoader(
    labeled_set,
    batch_size=flags.labeled_batch_size,
    num_workers=flags.num_workers,
    pin_memory=True,
    sampler=samplers.BalancedDatasetSampler(
                labeled_set,
                labeled_set.get_label,
                num_samples=flags.num_dataset_labeled),
    drop_last=True)
# all samples are unlabeled. A balanced sample is applied to these samples.
unlabeled_set = dataset.copy()
unlabeled_loader = torch.utils.data.DataLoader(
    unlabeled_set,
    batch_size=flags.unlabeled_batch_size,
    num_workers=flags.num_workers,
    pin_memory=True,
    sampler=samplers.BalancedDatasetSampler(
                unlabeled_set,
                unlabeled_set.get_label,
                num_samples=flags.num_dataset_unlabeled),
    drop_last=True)
# 30% of all samples are test data.
test_set, _ = dataset.split(0.3)
test_set.transform_fn = transform_fn
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=flags.test_batch_size,
    num_workers=flags.num_workers,
    pin_memory=True,
    shuffle=False)

print('len(dataset): ', len(dataset))
print('len(labeled_set): ', len(labeled_set))
print('len(unlabeled_set): ', len(unlabeled_set))
print('len(test_set): ', len(test_set))

path_to_model = args.path_to_model
outdir = args.path_to_outdir or flags.outdir
eval_step = args.eval_step or flags.eval_step
in_channels = len(flags.use_channels)

target_dict = {}
for i in range(len(labeled_set)):
    idx, name = labeled_set.get_label(i)
    target_dict[idx] = acronym(name)
print('target_dict:', target_dict)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
model = models.IIC(flags.model, in_channels=in_channels,
                   num_classes=flags.num_classes,
                   num_classes_over=flags.num_classes_over,
                   num_heads=flags.num_heads).to(device)
optimizer = get_optimizer(model, flags.optimizer, lr=flags.lr, weight_decay=flags.weight_decay)
model, optimizer = amp.initialize(model, optimizer, opt_level=flags.opt_level)

if os.path.exists(path_to_model or ''):
    model.load_part_of_state_dict(torch.load(path_to_model))

log = defaultdict(lambda: [])

def mutual_info_heads(y_outputs, yt_outputs, e=1e-8):

    def criterion(z, zt):
        _, k = z.size()
        p = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
        p = ((p + p.t()) / 2) / p.sum()
        p[(p < e).data] = e
        pi = p.sum(dim=1).view(k, 1).expand(k, k)
        pj = p.sum(dim=0).view(1, k).expand(k, k)
        return (p * (torch.log(pi) + torch.log(pj) - torch.log(p))).sum()

    loss_heads = []
    for i in range(flags.num_heads):
        y = y_outputs[i]
        yt = yt_outputs[i]
        tmp = criterion(y, yt)
        loss_heads.append(tmp)
    loss_heads = torch.stack(loss_heads)
    return loss_heads

def cross_entropy_heads(y_outputs, target):
    loss_heads = []
    for i in range(flags.num_heads):
        y = y_outputs[i]
        tmp = F.cross_entropy(y, target, weight=None, reduction='sum')
        loss_heads.append(tmp)
    loss_heads = torch.stack(loss_heads)
    return loss_heads

eps = torch.finfo(torch.half).eps
for epoch in range(1, flags.num_epochs):
    print(f'---------- epoch {epoch} ----------')
    model.train()
    if flags.reinitialize_headers_weights:
        model.initialize_headers_weights()
    loss = defaultdict(lambda: 0)
    head_selecter = torch.zeros(flags.num_heads).to(device)
    best_head_indices = []
    with tqdm(total=N_STEP) as pbar:
        for labeled_data, unlabeled_data in zip(cycle(labeled_loader), unlabeled_loader):
            # labeled loss
            x, xt, target = labeled_data
            x, xt = x.to(device, non_blocking=True), xt.to(device, non_blocking=True)
            target = target['target_index'].to(device, non_blocking=True)
            # labeled iic loss
            y_outputs, y_over_outputs = model(x)
            yt_outputs, yt_over_outputs = model(xt)
            loss_iic_labeled = mutual_info_heads(y_outputs, yt_outputs, eps) \
                + mutual_info_heads(y_over_outputs, yt_over_outputs, eps)
            loss_iic_labeled = loss_iic_labeled.mean()

            # labeled supervised loss
            loss_supervised = cross_entropy_heads(y_outputs, target) \
                + cross_entropy_heads(yt_outputs, target)
            best_head_idx = torch.argmin(loss_supervised, -1).item()
            best_head_indices.append(best_head_idx)
            loss_supervised = loss_supervised.mean()

            # unlabeled loss
            x, xt, _ = unlabeled_data
            x, xt = x.to(device, non_blocking=True), xt.to(device, non_blocking=True)
            # unlabeled iic loss
            y_outputs, y_over_outputs = model(x)
            yt_outputs, yt_over_outputs = model(xt)
            loss_iic_unlabeled = mutual_info_heads(y_outputs, yt_outputs, eps) \
                + mutual_info_heads(y_over_outputs, yt_over_outputs, eps)
            loss_iic_unlabeled = loss_iic_unlabeled.mean()

            loss_step = loss_iic_labeled * flags.weights[0] \
                        + loss_supervised * flags.weights[1] \
                        + loss_iic_unlabeled * flags.weights[2]

            loss["loss_iic_labeled"] += loss_iic_labeled.item()
            loss["loss_supervised"] += loss_supervised.item()
            loss["loss_iic_unlabeled"] += loss_iic_unlabeled.item()
            loss["loss_step"] += loss_step.item()

            optimizer.zero_grad()
            with amp.scale_loss(loss_step, optimizer) as loss_scaled:
                loss_scaled.backward()
            optimizer.step()

            pbar.update(1)

    # scheduler.step()
    for k, v in loss.items():
        log[k].append(v)

    print(f'loss_step: {loss["loss_step"]:.3f}')
    print(f'loss_iic_labeled: {loss["loss_iic_labeled"]:.3f}')
    print(f'loss_supervised: {loss["loss_supervised"]:.3f}')
    print(f'loss_iic_unlabeled: {loss["loss_iic_unlabeled"]:.3f}')
    print(f'best_head_indices: {best_head_indices}')

    if epoch % eval_step != 0:
        continue
    best_head_idx = Counter(best_head_indices).most_common()[0][0]

    model.eval()
    result = defaultdict(lambda: [])
    with torch.no_grad():
        for x, _, target in test_loader:
            x = x.to(device)
            target = target['target_index']
            y, y_over = model(x, head_index=best_head_idx)
            result['pred'].append(y.argmax(dim=-1).cpu())
            result['pred_over'].append(y_over.argmax(dim=-1).cpu())
            result['true'].append(target)
    preds = torch.cat(result['pred']).numpy()
    preds_over = torch.cat(result['pred_over']).numpy()
    trues = torch.cat(result['true']).numpy()

    # write result
    os.makedirs(outdir, exist_ok=True)
    cm_ylabels = [f'{i}-{target_dict[i]}' for i in range(max(trues)+1)]
    cm = np.zeros((flags.num_classes, max(trues) + 1), dtype=np.int)
    for i, j in zip(preds, trues):
        cm[i, j] += 1
    plot_cm(cm, list(range(flags.num_classes)), cm_ylabels,
            f'{outdir}/cm_{epoch}.png')
    cm_over = np.zeros((flags.num_classes_over, max(trues) + 1), dtype=np.int)
    for i, j in zip(preds_over, trues):
        cm_over[i, j] += 1
    plot_cm(cm_over, list(range(flags.num_classes_over)), cm_ylabels,
            f'{outdir}/cm_over_{epoch}.png')
    logger(log, epoch, outdir)
