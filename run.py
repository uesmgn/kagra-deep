import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import abc
import numbers
from tqdm import tqdm

from src.utils import transforms
from src.data import datasets
from src.data import samplers
from src import config

from mnist import M4

import matplotlib.pyplot as plt

# def wandb_init(args):
#     wandb.init(project=args.project, tags=args.tags, group=args.group)
#     wandb.run.name = args.name + "_" + wandb.run.id
#


@hydra.main(config_path="config", config_name="test")
def main(args):
    # wandb_init(args.wandb)

    transform_fn = transforms.Compose(
        [
            transforms.SelectIndices(args.use_channels, 0),
            transforms.CenterMaximizedResizeCrop(224),
        ]
    )
    augment_fn = transforms.Compose(
        [
            transforms.SelectIndices(args.use_channels, 0),
            transforms.RandomMaximizedResizeCrop(224),
        ]
    )
    alt = -1 if args.use_other else None
    target_transform_fn = transforms.ToIndex(args.targets, alt=alt)

    dataset = datasets.HDF5_C(args.dataset_root, transform_fn, augment_fn, target_transform_fn)
    train_set, test_set = dataset.split(train_size=args.train_size, stratify=dataset.targets)
    train_sampler = samplers.Balancer(train_set, args.batch_size * args.train_steps)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    model = M4(dim_x=64, dim_y=20, dim_y_over=30, dim_z=2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.num_epochs):
        model.train()
        total = 0
        for i, (x, xt, _) in enumerate(train_loader):
            x = x.to(device)
            xt = xt.to(device)
            loss = model(x, xt, mode="vae", weights=None)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.total.item()
        print("loss: {:.3f} at epoch: {}".format(total, epoch))

        model.eval()
        params = defaultdict(lambda: torch.tensor([]))
        with torch.no_grad():

            for i, (x, _, y) in enumerate(test_loader):
                x = x.to(device)
                z, qy, qy_over = model.params(x)
                y_pred = torch.argmax(qy, -1)
                y_over_pred = torch.argmax(qy_over, -1)

                params["z"] = torch.cat([params["z"], z.cpu()])
                params["y"] = torch.cat([params["y"], y])
                params["y_pred"] = torch.cat([params["y_pred"], y_pred.cpu()])
                params["y_over_pred"] = torch.cat([params["y_over_pred"], y_over_pred.cpu()])

            z = params["z"].numpy()
            y = params["y"].numpy().astype(int)
            y_pred = params["y_pred"].numpy().astype(int)
            y_over_pred = params["y_over_pred"].numpy().astype(int)

            plt.figure(figsize=(12, 12))
            for i in np.unique(y):
                idx = np.where(y == i)
                plt.scatter(z[idx, 0], z[idx, 1], label=i)
            plt.legend()
            plt.title(f"z_true_{epoch}")
            plt.savefig(f"z_true_{epoch}.png")
            plt.close()

            plt.figure(figsize=(12, 12))
            for i in np.unique(y_pred):
                idx = np.where(y_pred == i)
                plt.scatter(z[idx, 0], z[idx, 1], label=i)
            plt.legend()
            plt.title(f"z_pred_{epoch}")
            plt.savefig(f"z_pred_{epoch}.png")
            plt.close()

    #
    # cfg_params = {
    #     "type": args.type,
    #     "transform": transform_fn,
    #     "augment_transform": augment_fn,
    #     "target_transform": target_transform_fn,
    #     "sampler_callback": sampler_callback,
    #     "train_size": args.train_size,
    #     "labeled_size": args.labeled_size,
    # }
    #
    # cfg = config.Config(**cfg_params)
    #
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # if torch.cuda.is_available():
    #     torch.backends.cudnn.benchmark = True
    #
    # num_epochs = args.num_epochs
    #
    # model = M4(transform_fn, augment_fn, dim_x=64, dim_y=20, dim_y_over=30, dim_z=2).to(device)
    # optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=2, T_mult=2)
    #
    # train_set, eval_set = cfg.get_datasets(args.dataset)
    # train_loader = cfg.get_loader(args.train, train_set)
    # eval_loader = cfg.get_loader(args.eval, eval_set, train=False)
    #
    # for epoch in range(args.num_epochs):
    #     model.train()
    #     total = 0
    #     for (ux, uxt, _), (lx, _, target) in train_loader:
    #         ux = ux.to(device)
    #         uxt = uxt.to(device)
    #         lx = lx.to(device)
    #         target = target.to(device)
    #         y = F.one_hot(target, num_classes=args.num_classes)
    #
    #         loss = model(ux, xt=uxt) + model(lx, y=y)
    #
    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()
    #
    #         total += loss.item()
    #     print("loss: {:.3f} at epoch: {}".format(total, epoch))
    #     scheduler.step()


if __name__ == "__main__":
    main()
