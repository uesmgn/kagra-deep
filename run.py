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

from src import flatten, get_net, get_model, get_optim, get_dataset, get_sampler, get_loader
from src import utils


def wandb_init(args):
    wandb.init(project=args.project,
               tags=args.tags,
               group=args.group)
    wandb.run.name = args.name + '_' + wandb.run.id


def train(model, device, trainer, optim, epoch, use_amp=False):
    print(f"----- train at epoch: {epoch} -----")
    model.train()
    loss, num_samples = 0, 0
    with tqdm(total=len(trainer)) as pbar:
        for step, (x, target) in enumerate(trainer):
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            _, loss_step = model(x, target)
            optim.zero_grad()
            if use_amp:
                with amp.scale_loss(loss_step, optim) as loss_scaled:
                    loss_scaled.backward()
            else:
                loss_step.backward()
            optim.step()
            loss += loss_step.item()
            num_samples += x.shape[0]
            pbar.update(1)
    loss /= num_samples
    wandb.log({"epoch": epoch, "loss_train": loss})

def wandb_log(data, name, targets, type=None):
    import math
    import matplotlib.pyplot as plt
    import torchvision as tv
    from sklearn.manifold import TSNE

    labels = np.unique(targets)
    indices = {t: np.where(targets==t)[0] for t in np.unique(targets)}

    if type == "grid_image" and data.ndim == 3:
        idx = [v[0] for v in indices.values()]
        nrow = math.ceil(np.sqrt(len(idx))) + 1
        grid = tv.utils.make_grid(data[idx, ...], nrow=nrow).permute(1,2,0)
        plt.imshow(grid)
        plt.axis("off")
        plt.tight_layout()
        wandb.log({"epoch": epoch, name: plt})
        plt.close()
    elif type == "tsne" and data.ndim == 2:
        z = TSNE(n_components=2).fit_transform(data)
        xx, yy = z.T
        for target in targets:
            idx = np.where(labels==target)
            x = xx[idx]
            y = yy[idx]
            plt.scatter(x, y, s=8.0, label=target)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                   borderaxespad=0, fontsize=8)
        plt.tight_layout()
        wandb.log({"epoch": epoch, name: plt})
        plt.close()


def test(model, device, tester, epoch, log_params={}):
    print(f"----- test at epoch: {epoch} -----")
    model.eval()
    loss, num_samples = 0, 0
    targets = []
    logger = defaultdict(lambda: [])
    with torch.no_grad():
        with tqdm(total=len(tester)) as pbar:
            for step, (x, target) in enumerate(tester):
                x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
                params, loss_step = model(x, target)
                targets.append(target)
                for key, value in params.items():
                    if key in log_params and torch.is_tensor(value):
                        logger[key].append(value.squeeze())

                loss += loss_step.item()
                num_samples += x.shape[0]
                pbar.update(1)
    loss /= num_samples
    targets = torch.cat(targets).squeeze().cpu()

    wandb.log({"epoch": epoch, "loss_test": loss})
    for key, params in log_params.items():
        data = torch.cat(logger[key]).squeeze().cpu()
        wandb_log(data, key, targets, **params)



def run(args):
    wandb.config.update(flatten(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

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
        if epoch % args.eval_step == args.eval_step - 1:
            test(model, device, tester, epoch, log_params=args.log_params)


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    wandb_init(cfg.wandb)
    run(cfg.run)


if __name__ == "__main__":
    main()
