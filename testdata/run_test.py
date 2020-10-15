import wandb
import hydra
import random
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torchvision.transforms as tf
try:
    from apex import amp
except ImportError:
    raise ImportError('Please install apex using...')

from src.utils import config, stats, transforms
from src.utils import functional as F

# -------------------------------------------
# Implementation of Variational Autoencoder
# -------------------------------------------

def wandb_init(args):
    wandb.init(project=args.project,
               tags=args.tags,
               group=args.group)
    wandb.run.name = args.name + '_' + wandb.run.id

def run(args):
    wandb.config.update(F.flatten(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    config =

    net = get_net(args.net.name, **args.net.params)

    model = get_model(args.model.name, net=net, **args.model.params).to(device)
    optim = get_optim(model.parameters(), args.optim.name, **args.optim.params)
    if args.use_amp:
        model, optim = amp.initialize(model, optim, opt_level=args.opt_level)

    alt = -1 if args.use_other else None
    target_transform = utils.transforms.ToIndex(args.targets, alt)

    dataset = get_dataset(args.dataset.name, **args.dataset.params,
                          target_transform=target_transform)

    transform = tf.Compose([
        utils.transforms.SelectIndices(args.use_channels, 0),
        utils.transforms.CenterMaximizedResizeCrop(224),
    ])
    augment = tf.Compose([
        utils.transforms.SelectIndices(args.use_channels, 0),
        utils.transforms.RandomMaximizedResizeCrop(224),
    ])

    train_set, test_set = dataset.split(
        args.train_size, stratify=dataset.targets)
    train_set.transform = augment
    test_set.transform = transform

    sampler = get_sampler(args.sampler.name, dataset=train_set, **args.sampler.params)

    trainer = get_loader(train_set, sampler=sampler,
                         drop_last=True, **args.loader.params)
    tester = get_loader(test_set, **args.loader.params)

    for epoch in range(args.num_epochs):
        train(model, device, trainer, optim, epoch, use_amp=args.use_amp)
        if epoch % args.eval_step == 0:
            test(model, device, tester, epoch, log_params=args.log_params)

@hydra.main(config_path="config", config_name="vae")
def main(cfg):
    wandb_init(cfg.wandb)
    run(cfg.run)

if __name__ == "__main__":
    main()
