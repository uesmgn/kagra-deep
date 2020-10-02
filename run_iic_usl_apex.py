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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
    perturb = torchvision.transforms.Lambda(lambda x: (transform(x), augment(x)))
    dataset = datasets.HDF5Dataset(cfg.dataset, transform=perturb)
    sample, _, _ = dataset[0]
    train_set, test_set = dataset.split(cfg.rate_train)
    K = len(dataset.targets)
    labels = [to_acronym(l) for l in dataset.targets]
    K_over = cfg.iic.num_classes_over
    train_sampler = None
    if cfg.balancing:
        train_sampler = samplers.BalancingSampler(train_set, train_set.get_label,
                                                  num_samples=cfg.num_balancing_samples)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,
                                               pin_memory=cfg.pin_memory,
                                               sampler=train_sampler,
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
    if cfg.use_amp:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Please install apex using...')
        model, optim = amp.initialize(model, optim, opt_level=cfg.opt_level)

    logger = stats.EpochLogger()
    for epoch in range(1, cfg.num_epochs+1):
        print(f'---------- epoch {epoch} ----------')
        model.train()
        loss_train = 0
        head_selector = 0
        head_selector_over = 0
        if cfg.iic.reinitialize_header_weights:
            model.initialize_headers_weights()
        with tqdm(total=cfg.batch_size * len(train_loader)) as pbar:
            for x, xt, target in train_loader:
                x, xt = x.to(device, non_blocking=True), xt.to(device, non_blocking=True)
                y, y_over = model(x)
                yt, yt_over = model(xt)
                crit_y = model.crit(y, yt, 0)
                crit_y_over = model.crit(y_over, yt_over, 0)
                crit = crit_y + crit_y_over
                loss_batch = crit.sum()
                optim.zero_grad()
                if cfg.use_amp:
                    with amp.scale_loss(loss_batch, optim) as loss_scaled:
                        loss_scaled.backward()
                else:
                    loss_batch.backward()
                optim.step()
                loss_train += loss_batch.item()
                head_selector += crit_y
                head_selector_over += crit_y_over
                pbar.update(x.shape[0])
        best_idx = torch.argmin(head_selector, -1).item()
        best_idx_over = torch.argmin(head_selector_over, -1).item()
        logger.update('loss_train', loss_train, verbose=True)
        logger.update('best_idx', best_idx, verbose=True)
        logger.update('best_idx_over', best_idx_over, verbose=True)

        if epoch % cfg.checkpoint.eval == 0:
            evaluator = stats.Evaluator()
            model.eval()
            with torch.no_grad():
                with tqdm(total=cfg.batch_size * len(test_loader)) as pbar:
                    for x, _, target in test_loader:
                        x = x.to(device, non_blocking=True)
                        y, y_over = model(x)
                        y, y_over = y[best_idx], y_over[best_idx_over]
                        evaluator.update('target', target)
                        evaluator.update('y', y.argmax(dim=-1))
                        evaluator.update('y_over', y_over.argmax(dim=-1))
                        pbar.update(x.shape[0])
            fig, ax = evaluator.get_confusion_matrix('y', 'target', K, labels)
            fig.savefig(f'cm_{epoch}.png')
            plt.close(fig)
            fig, ax = evaluator.get_confusion_matrix('y_over', 'target', K_over, labels)
            fig.savefig(f'cm_over_{epoch}.png')
            plt.close(fig)
            for k, fig, ax in logger.get_plots():
                fig.savefig(f'{k}_{epoch}.png')
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






    # with tqdm(total=cfg.batch_size * len(train_loader)) as pbar:
    #     for x, xt, target in train_loader:
    #






if __name__ == "__main__":
    run()


#
# transform_fn = torchvision.transforms.Compose([
#     torchvision.transforms.Lambda(lambda x: torch.stack([x[i] for i in flags.use_channels])),
#     torchvision.transforms.ToPILImage(),
#     torchvision.transforms.CenterCrop((479, 479)),
#     torchvision.transforms.Resize((224, 224)),
#     torchvision.transforms.ToTensor(),
# ])
#
# argmentation_fn = torchvision.transforms.Compose([
#     torchvision.transforms.Lambda(lambda x: torch.stack([x[i] for i in flags.use_channels])),
#     torchvision.transforms.ToPILImage(),
#     torchvision.transforms.RandomCrop((479, 479)),
#     torchvision.transforms.Resize((224, 224)),
#     torchvision.transforms.ToTensor(),
# ])
#
# perturb_fn = torchvision.transforms.Compose([
#     torchvision.transforms.Lambda(lambda x: (transform_fn(x), argmentation_fn(x)))
# ])
#
# parser = argparse.ArgumentParser()
# parser.add_argument('path_to_hdf', type=validation.is_hdf,
#                     help='path to hdf file.')
# parser.add_argument('-m', '--path_to_model', type=validation.is_file,
#                     help='path to pre-trained model.state_dict().')
# parser.add_argument('-o', '--path_to_outdir', type=validation.is_dir,
#                     help='path to output directory.')
# parser.add_argument('-e', '--eval_step', type=int,
#                     help='evaluating step.')
# parser.add_argument('-n', '--num_per_label', type=int,
#                     help='num of labeled samples per label.')
# args = parser.parse_args()
#
# num_per_label = args.num_per_label or flags.num_per_label
#
# dataset = datasets.HDF5Dataset(args.path_to_hdf,
#                 data_shape=flags.input_shape,
#                 transform_fn=perturb_fn)
# # random num_per_label samples of each label are labeled.
# labeled_set, _ = dataset.split_balanced('target_index', num_per_label)
# labeled_loader = torch.utils.data.DataLoader(
#     labeled_set,
#     batch_size=flags.labeled_batch_size,
#     num_workers=flags.num_workers,
#     pin_memory=True,
#     sampler=samplers.BalancedDatasetSampler(
#                 labeled_set,
#                 labeled_set.get_label,
#                 num_samples=flags.num_dataset_labeled),
#     drop_last=True)
# # all samples are unlabeled. A balanced sample is applied to these samples.
# unlabeled_set = dataset.copy()
# unlabeled_loader = torch.utils.data.DataLoader(
#     unlabeled_set,
#     batch_size=flags.unlabeled_batch_size,
#     num_workers=flags.num_workers,
#     pin_memory=True,
#     sampler=samplers.BalancedDatasetSampler(
#                 unlabeled_set,
#                 unlabeled_set.get_label,
#                 num_samples=flags.num_dataset_unlabeled),
#     drop_last=True)
# # 30% of all samples are test data.
# test_set, _ = dataset.split(0.3)
# test_set.transform_fn = transform_fn
# test_loader = torch.utils.data.DataLoader(
#     test_set,
#     batch_size=flags.test_batch_size,
#     num_workers=flags.num_workers,
#     pin_memory=True,
#     shuffle=False)
#
# print('len(dataset): ', len(dataset))
# print('len(labeled_set): ', len(labeled_set))
# print('len(unlabeled_set): ', len(unlabeled_set))
# print('len(test_set): ', len(test_set))
#
# path_to_model = args.path_to_model
# outdir = args.path_to_outdir or flags.outdir
# eval_step = args.eval_step or flags.eval_step
# in_channels = len(flags.use_channels)
# model_file = args.path_to_model or flags.path_to_model
#
# target_dict = {}
# for i in range(len(labeled_set)):
#     idx, name = labeled_set.get_label(i)
#     target_dict[idx] = acronym(name)
# print('target_dict:', target_dict)
#
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True
# model = models.IIC(flags.model, in_channels=in_channels,
#                    num_classes=flags.num_classes,
#                    num_classes_over=flags.num_classes_over,
#                    num_heads=flags.num_heads).to(device)
# optimizer = get_optimizer(model, flags.optimizer, lr=flags.lr, weight_decay=flags.weight_decay)
# model, optimizer = amp.initialize(model, optimizer, opt_level=flags.opt_level)
#
# if os.path.exists(path_to_model or ''):
#     model.load_part_of_state_dict(torch.load(path_to_model))
#
# log = defaultdict(lambda: [])
#
# def mutual_info_heads(y_outputs, yt_outputs, e=1e-8):
#
#     def criterion(z, zt):
#         _, k = z.size()
#         p = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
#         p = ((p + p.t()) / 2) / p.sum()
#         p[(p < e).data] = e
#         pi = p.sum(dim=1).view(k, 1).expand(k, k)
#         pj = p.sum(dim=0).view(1, k).expand(k, k)
#         return (p * (torch.log(pi) + torch.log(pj) - torch.log(p))).sum()
#
#     loss_heads = []
#     for i in range(flags.num_heads):
#         y = y_outputs[i]
#         yt = yt_outputs[i]
#         tmp = criterion(y, yt)
#         loss_heads.append(tmp)
#     loss_heads = torch.stack(loss_heads)
#     return loss_heads
#
# eps = torch.finfo(torch.float).eps
# for epoch in range(1, flags.num_epochs):
#     print(f'---------- epoch {epoch} ----------')
#     model.train()
#     if flags.reinitialize_headers_weights:
#         model.initialize_headers_weights()
#     loss = defaultdict(lambda: 0)
#     head_selecter = torch.zeros(flags.num_heads).to(device)
#     best_head_indices = []
#     with tqdm(total=N_STEP) as pbar:
#         for unlabeled_data in zip(unlabeled_loader):
#             # unlabeled loss
#             x, xt, _ = unlabeled_data
#             x, xt = x.to(device, non_blocking=True), xt.to(device, non_blocking=True)
#             # unlabeled iic loss
#             y_outputs, y_over_outputs = model(x)
#             yt_outputs, yt_over_outputs = model(xt)
#             loss_iic_unlabeled = mutual_info_heads(y_outputs, yt_outputs, eps) \
#                 + mutual_info_heads(y_over_outputs, yt_over_outputs, eps)
#             loss_iic_unlabeled = loss_iic_unlabeled.mean()
#
#             loss_step = loss_iic_unlabeled
#
#             loss["loss_step"] += loss_step.item()
#
#             optimizer.zero_grad()
#             with amp.scale_loss(loss_step, optimizer) as loss_scaled:
#                 loss_scaled.backward()
#             optimizer.step()
#
#             pbar.update(1)
#
#     # scheduler.step()
#     for k, v in loss.items():
#         log[k].append(v)
#
#     print(f'loss_step: {loss["loss_step"]:.3f}')
#     print(f'loss_iic_labeled: {loss["loss_iic_labeled"]:.3f}')
#     print(f'loss_supervised: {loss["loss_supervised"]:.3f}')
#     print(f'loss_iic_unlabeled: {loss["loss_iic_unlabeled"]:.3f}')
#     print(f'best_head_indices: {best_head_indices}')
#
#     if epoch % eval_step != 0:
#         continue
#     best_head_idx = Counter(best_head_indices).most_common()[0][0]
#
#     model.eval()
#     result = defaultdict(lambda: [])
#     with torch.no_grad():
#         for x, target in test_loader:
#             x = x.to(device)
#             target = target['target_index']
#             y, y_over = model(x, head_index=best_head_idx)
#             result['pred'].append(y.argmax(dim=-1).cpu())
#             result['pred_over'].append(y_over.argmax(dim=-1).cpu())
#             result['true'].append(target)
#     preds = torch.cat(result['pred']).numpy()
#     preds_over = torch.cat(result['pred_over']).numpy()
#     trues = torch.cat(result['true']).numpy()
#
#     # write result
#     os.makedirs(outdir, exist_ok=True)
#     cm_ylabels = [f'{i}-{target_dict[i]}' for i in range(max(trues)+1)]
#     cm = np.zeros((flags.num_classes, max(trues) + 1), dtype=np.int)
#     for i, j in zip(preds, trues):
#         cm[i, j] += 1
#     plot_cm(cm, list(range(flags.num_classes)), cm_ylabels,
#             f'{outdir}/cm_{epoch}.png')
#     cm_over = np.zeros((flags.num_classes_over, max(trues) + 1), dtype=np.int)
#     for i, j in zip(preds_over, trues):
#         cm_over[i, j] += 1
#     plot_cm(cm_over, list(range(flags.num_classes_over)), cm_ylabels,
#             f'{outdir}/cm_over_{epoch}.png')
#     logger(log, epoch, outdir)
#
#     if epoch % flags.save_step == 0:
#         print(f'---------- saving check point at epoch {epoch} ----------')
#         torch.save(model.state_dict(), os.path.join(outdir, model_file))
