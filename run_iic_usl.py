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
import hydra
from omegaconf import DictConfig

from src.utils import validation, config, stats
import src.models as models
from src.data import datasets, samplers

SEED_VALUE = 1234
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)

def to_acronym(name):
    return re.sub(r'(^[0-9a-zA-Z]{5,}(?=_))|((?<=_)[0-9a-zA-Z]*)',
                  lambda m: str(m.group(1) or '')[
                      :3] + str(m.group(2) or '')[:1],
                  name).replace('_', '.')


@hydra.main(config_path="config", config_name="config")
def run(cfg: DictConfig):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    c, h, w = cfg.dataset.shape
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.CenterCrop((h, h)),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
    augment = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomCrop((h, h)),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
    perturb = torchvision.transforms.Lambda(
        lambda x: (transform(x), augment(x)))
    dataset = datasets.HDF5Dataset(cfg.dataset, transform=perturb)
    sample, _, _ = dataset[0]
    train_set, test_set = dataset.split(cfg.rate_train)
    print('train_set:', train_set.counter)
    print('test_set:', test_set.counter)
    K = len(dataset.targets)
    labels = [to_acronym(l) for l in dataset.targets]
    K_over = cfg.iic.num_classes_over
    balancer = None
    if cfg.balancing:
        balancer = samplers.Balancer(train_set, train_set.get_label,
                                     num_samples=cfg.num_balancing_samples)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,
                                               pin_memory=cfg.pin_memory,
                                               sampler=balancer,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=cfg.batch_size,
                                              num_workers=cfg.num_workers,
                                              pin_memory=cfg.pin_memory,
                                              drop_last=False)

    model = models.IIC(cfg.model.name, in_channels=sample.shape[0],
                       num_classes=len(dataset.targets),
                       num_classes_over=cfg.iic.num_classes_over,
                       num_heads=cfg.iic.num_heads).to(device)
    optim = config.get_optim(model.parameters(), cfg.optim)
    weights = cfg.iic.usl.weights
    if cfg.use_amp:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Please install apex using...')
        model, optim = amp.initialize(model, optim, opt_level=cfg.opt_level)

    logger = stats.EpochLogger()
    for epoch in range(1, cfg.num_epochs + 1):
        print(f'---------- epoch {epoch} ----------')
        model.train()
        loss = defaultdict(lambda: 0)
        head_selector = 0
        head_selector_over = 0
        if cfg.iic.reinitialize_header_weights:
            model.initialize_headers_weights()
        with tqdm(total=cfg.batch_size * len(train_loader)) as pbar:
            for x, xt, target in train_loader:
                x, xt = x.to(device, non_blocking=True), xt.to(
                    device, non_blocking=True)
                y, y_over = model(x)
                yt, yt_over = model(xt)
                mi = model.mutual_info(y, yt, 0) * weights.mi
                head_selector += mi
                loss['loss_mi'] += mi.sum().item()
                mi_over = model.mutual_info(y_over, yt_over, 0) * weights.mi_over
                head_selector_over += mi_over
                loss['loss_mi_over'] += mi_over.sum().item()
                loss_batch = mi.sum() + mi_over.sum()
                loss['loss_train'] += loss_batch.item()
                optim.zero_grad()
                if cfg.use_amp:
                    with amp.scale_loss(loss_batch, optim) as loss_scaled:
                        loss_scaled.backward()
                else:
                    loss_batch.backward()
                optim.step()
                pbar.update(x.shape[0])
        best_idx = torch.argmin(head_selector, -1).item()
        best_idx_over = torch.argmin(head_selector_over, -1).item()
        for key, value in loss.items():
            logger.update(key, value, verbose=True)
        logger.update('best_idx', best_idx, verbose=True)
        logger.update('best_idx_over', best_idx_over, verbose=True)

        if epoch % cfg.checkpoint.eval == 0:
            evaluator = stats.Evaluator()
            model.eval()
            with torch.no_grad():
                with tqdm(total=len(test_set)) as pbar:
                    for x, _, target in test_loader:
                        x = x.to(device, non_blocking=True)
                        y, y_over = model(x)
                        y, y_over = y[best_idx], y_over[best_idx_over]
                        evaluator.update('target', target)
                        evaluator.update('y', y.argmax(dim=-1))
                        evaluator.update('y_over', y_over.argmax(dim=-1))
                        pbar.update(x.shape[0])
            fig, ax = evaluator.get_confusion_matrix('y', 'target', K, labels)
            fig.savefig(f'iic_usl_cm_{epoch}.png')
            plt.close(fig)
            fig, ax = evaluator.get_confusion_matrix(
                'y_over', 'target', K_over, labels)
            fig.savefig(f'iic_usl_cm_over_{epoch}.png')
            plt.close(fig)
            for k, fig, ax in logger.get_plots():
                fig.savefig(f'iic_usl_{k}_{epoch}.png')
                plt.close(fig)

        if epoch % cfg.checkpoint.save == 0:
            state_dicts = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'epoch': epoch
            }
            if cfg.use_amp:
                state_dicts['amp'] = amp.state_dict()
            torch.save(state_dicts, 'iic_usl.pt')


if __name__ == "__main__":
    run()
