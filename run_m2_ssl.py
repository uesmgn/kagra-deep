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

from mnist import M2

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
    labeled_set, unlabeled_set = train_set.split(train_size=args.labeled_size, stratify=train_set.targets)
    print("labeled_set:", labeled_set.counter)
    print("unlabeled_set:", unlabeled_set.counter)
    print("test_set:", test_set.counter)
    labeled_set.transform, unlabeled_set.transform = augment_fn, augment_fn

    def sampler_callback(ds, batch_size):
        # return samplers.Upsampler(ds, batch_size * args.num_train_steps)
        return samplers.Balancer(ds, batch_size * args.num_train_steps)

    labeled_loader = torch.utils.data.DataLoader(
        labeled_set,
        batch_size=16,
        num_workers=args.num_workers,
        sampler=sampler_callback(labeled_set, 16),
        pin_memory=True,
        drop_last=True,
    )
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler_callback(unlabeled_set, args.batch_size),
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
    model = M2(ch_in=args.ch_in, dim_y=args.num_classes, dim_z=args.dim_z).to(device)
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
        for (ux, _), (lx, y) in zip(unlabeled_loader, cycle(labeled_loader)):
            ux = ux.to(device)
            lx = lx.to(device)
            y = y.to(device)
            bce, kl_gauss, kl_cat = model(ux)
            bce_, ce = model(lx, y)
            bce += bce_
            loss = bce + kl_gauss + kl_cat + 1000.0 * ce
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            total_dict["total"] += loss.item()
            total_dict["bce"] += bce.item()
            total_dict["kl_gauss"] += kl_gauss.item()
            total_dict["kl_cat"] += kl_cat.item()
            total_dict["ce"] += ce.item()
        for key, value in total_dict.items():
            print("loss_{}: {:.3f} at epoch: {}".format(key, value, epoch))
            stats[key].append(value)

        scheduler.step()

        if epoch % args.eval_interval == 0:
            print(f"----- evaluating at epoch {epoch} -----")
            model.eval()
            params = defaultdict(lambda: torch.tensor([]))

            with torch.no_grad():
                for i, (x, y) in enumerate(test_loader):
                    x = x.to(device)
                    qy, qy_pi = model.qy_x(x, hard=True)
                    _, qz, _ = model.qz_xy(x, qy)
                    y_pred = torch.argmax(qy_pi, -1)

                    params["qz"] = torch.cat([params["qz"], qz.cpu()])
                    params["y"] = torch.cat([params["y"], y])
                    params["y_pred"] = torch.cat([params["y_pred"], y_pred.cpu()])

                for i in range(args.num_classes):
                    y_i = F.one_hot(torch.full((1000,), i).long(), num_classes=args.num_classes)
                    pz, _, _ = model.pz_y(y_i.float().to(device))
                    params["pz"] = torch.cat([params["pz"], pz.cpu()])
                yy = torch.tensor(list(range(args.num_classes)))
                yy = yy.unsqueeze(1).repeat(1, 1000).flatten()

                qz = params["qz"].numpy()
                pz = params["pz"].numpy()
                umapper = umap.UMAP(min_dist=0.5, random_state=123).fit(qz)
                qz = umapper.embedding_
                pz = umapper.transform(pz)

                y = params["y"].numpy().astype(int)
                y_pred = params["y_pred"].numpy().astype(int)

                plt.figure(figsize=(12, 12))
                for i in np.unique(y):
                    idx = np.where(y == i)
                    c = colormap(i)
                    plt.scatter(qz[idx, 0], qz[idx, 1], c=c, label=targets[i], edgecolors=darken(c))
                plt.legend(loc="upper right")
                plt.title(f"qz_true at epoch {epoch}")
                plt.savefig(f"qz_true_{epoch}.png")
                plt.close()

                plt.figure(figsize=(12, 12))
                for i in np.unique(y_pred):
                    idx = np.where(y_pred == i)
                    c = colormap(i)
                    plt.scatter(qz[idx, 0], qz[idx, 1], c=c, label=targets[i], edgecolors=darken(c))
                plt.legend(loc="upper right")
                plt.title(f"qz_pred at epoch {epoch}")
                plt.savefig(f"qz_pred_{epoch}.png")
                plt.close()

                plt.figure(figsize=(12, 12))
                for i in np.unique(yy):
                    idx = np.where(yy == i)
                    c = colormap(i)
                    plt.scatter(pz[idx, 0], pz[idx, 1], c=c, label=targets[i], edgecolors=darken(c))
                plt.legend(loc="upper right")
                plt.title(f"pz at epoch {epoch}")
                plt.savefig(f"pz_{epoch}.png")
                plt.close()

                plt.figure(figsize=(20, 12))
                cm = confusion_matrix(y, y_pred)[: args.num_classes, :]
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, yticklabels=targets)
                plt.yticks(rotation=45)
                plt.title(f"confusion matrix y / y' at epoch {epoch}")
                plt.savefig(f"cm_y_{epoch}.png")
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
