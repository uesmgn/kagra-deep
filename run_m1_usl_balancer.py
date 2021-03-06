import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import abc, defaultdict
import numbers
from tqdm import tqdm
import umap
from itertools import cycle

from src.utils.functional import acronym, darken, colormap
from src.utils import transforms
from src.data import datasets
from src.data import samplers
from src import config

from mnist import M1

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


@hydra.main(config_path="config", config_name="test")
def main(args):

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

    targets = [acronym(target) for target in args.targets]

    dataset = datasets.HDF5(args.dataset_root, transform_fn, target_transform_fn)
    train_set, test_set = dataset.split(train_size=args.train_size, stratify=dataset.targets)
    train_set.transform = augment_fn

    def sampler_callback(ds, num_samples):
        return samplers.Balancer(ds, num_samples)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler_callback(train_set, args.batch_size * args.num_train_steps),
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
    model = M1(ch_in=args.ch_in, dim_z=args.dim_z).to(device)
    if args.load_state_dict:
        model.load_state_dict_part(torch.load(args.model_path))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)

    stats = defaultdict(lambda: [])

    for epoch in range(args.num_epochs):
        print(f"----- training at epoch {epoch} -----")
        model.train()
        total = 0
        total_dict = defaultdict(lambda: 0)
        for x, _ in tqdm(train_loader):
            x = x.to(device)
            bce, kl_gauss = model(x)
            loss = bce + kl_gauss
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            total_dict["total"] += loss.item()
            total_dict["bce"] += bce.item()
            total_dict["kl_gauss"] += kl_gauss.item()
        for key, value in total_dict.items():
            print("loss_{}: {:.3f} at epoch: {}".format(key, value, epoch))
            stats[key].append(value)

        scheduler.step()

        if epoch % args.eval_interval == 0:
            print(f"----- evaluating at epoch {epoch} -----")
            model.eval()
            params = defaultdict(lambda: torch.tensor([]))

            with torch.no_grad():
                for i, (x, y) in tqdm(enumerate(test_loader)):
                    x = x.to(device)
                    _, qz, _ = model.qz_x(x)

                    params["qz"] = torch.cat([params["qz"], qz.cpu()])
                    params["y"] = torch.cat([params["y"], y])

                qz = params["qz"].numpy()
                umapper = umap.UMAP(min_dist=0.5, random_state=123).fit(qz)
                qz = umapper.embedding_

                y = params["y"].numpy().astype(int)

                plt.figure(figsize=(12, 12))
                for i in np.unique(y):
                    idx = np.where(y == i)
                    c = colormap(i)
                    plt.scatter(qz[idx, 0], qz[idx, 1], c=c, label=targets[i], edgecolors=darken(c))
                plt.legend()
                plt.title(f"qz_true at epoch {epoch}")
                plt.savefig(f"qz_true_{epoch}.png")
                plt.close()

                if epoch > 0:
                    for key, value in stats.items():
                        plt.plot(value)
                        plt.ylabel(key)
                        plt.xlabel("epoch")
                        plt.title(key)
                        plt.xlim((0, len(value) - 1))
                        plt.savefig(f"loss_{key}_{epoch}.png")
                        plt.close()

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), args.model_path)


if __name__ == "__main__":
    main()
