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

from mnist import M3

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
    train_set = datasets.co(train_set, augment_fn)

    def sampler_callback(ds, batch_size):
        return samplers.Upsampler(ds, batch_size * args.num_train_steps)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler_callback(train_set, args.batch_size),
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
    model = M3(ch_in=args.ch_in, dim_y=args.num_classes, dim_w=args.dim_w, dim_z=args.dim_z).to(device)
    if args.load_state_dict:
        model.load_state_dict_part(torch.load(args.model_path))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=2, T_mult=2)
    thres = 0.8
    stats = defaultdict(lambda: [])

    for epoch in range(args.num_epochs):
        print(f"----- training at epoch {epoch} -----")
        model.train()
        total = 0
        total_dict = defaultdict(lambda: 0)
        for data, _ in tqdm(train_loader):
            x, v = data
            x = x.to(device)
            v = v.to(device)
            bce, kl_gauss = model.vae(x)
            bce_, kl_gauss_ = model.vae(v)
            bce += bce_
            kl_gauss += kl_gauss_
            mi_y, mi_w = model.iic(x, v)
            loss = bce + 10.0 * kl_gauss + 100.0 * mi_y + 100.0 * mi_w
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            total_dict["total"] += loss.item()
            total_dict["bce"] += bce.item()
            total_dict["kl_gauss"] += kl_gauss.item()
            total_dict["mi_y"] += mi_y.item()
            total_dict["mi_w"] += mi_w.item()
        for key, value in total_dict.items():
            print("loss_{}: {:.3f} at epoch: {}".format(key, value, epoch))
            stats[key].append(value)
        scheduler.step()

        if epoch % args.eval_interval == 0:
            print(f"----- evaluating at epoch {epoch} -----")
            model.eval()
            params = defaultdict(lambda: torch.tensor([]))
            indices = torch.tensor([]).long()
            with torch.no_grad():
                n = 0
                for i, (x, y) in tqdm(enumerate(test_loader)):
                    x = x.to(device)
                    qz, qy, qw = model.params(x)
                    idx = torch.nonzero(qy > thres).squeeze() + n
                    y_pred = torch.argmax(qy, -1)
                    w_pred = torch.argmax(qw, -1)

                    params["qz"] = torch.cat([params["qz"], qz.cpu()])
                    params["y"] = torch.cat([params["y"], y])
                    params["y_pred"] = torch.cat([params["y_pred"], y_pred.cpu()])
                    params["w_pred"] = torch.cat([params["w_pred"], w_pred.cpu()])
                    indices = torch.cat([indices, idx.cpu()])
                    n += x.shape[0]

                qz = params["qz"].numpy()
                umapper = umap.UMAP(min_dist=0.5, random_state=123).fit(qz)
                qz = umapper.embedding_

                y = params["y"].numpy().astype(int)
                y_pred = params["y_pred"].numpy().astype(int)
                w_pred = params["w_pred"].numpy().astype(int)
                indices = indices.numpy().astype(int)

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
                    plt.scatter(qz[idx, 0], qz[idx, 1], c=c, label=i, edgecolors=darken(c))
                plt.legend(loc="upper right")
                plt.title(f"qz_y at epoch {epoch}")
                plt.savefig(f"qz_y_{epoch}.png")
                plt.close()

                plt.figure(figsize=(12, 12))
                for i in np.unique(y_pred):
                    idx = np.where(y_pred[indices] == i)
                    c = colormap(i)
                    try:
                        plt.scatter(qz[indices][idx, 0], qz[indices][idx, 1], c=c, label=i, edgecolors=darken(c))
                    except:
                        pass
                plt.legend(loc="upper right")
                plt.title(f"qz_y filtered at epoch {epoch}")
                plt.savefig(f"qz_y_filtered_{epoch}.png")
                plt.close()

                plt.figure(figsize=(12, 12))
                for i in np.unique(w_pred):
                    idx = np.where(w_pred == i)
                    c = colormap(i)
                    plt.scatter(qz[idx, 0], qz[idx, 1], c=c, label=i, edgecolors=darken(c))
                plt.legend(loc="upper right")
                plt.title(f"qz_w at epoch {epoch}")
                plt.savefig(f"qz_w_{epoch}.png")
                plt.close()

                plt.figure(figsize=(20, 12))
                cm = confusion_matrix(y, y_pred)[: args.num_classes, :]
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, yticklabels=targets)
                plt.yticks(rotation=45)
                plt.title(f"confusion matrix y / y' at epoch {epoch}")
                plt.savefig(f"cm_y_{epoch}.png")
                plt.close()

                plt.figure(figsize=(20, 12))
                cm = confusion_matrix(y, w_pred)[: args.num_classes, :]
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, yticklabels=targets)
                plt.yticks(rotation=45)
                plt.title(f"confusion matrix y / w' at epoch {epoch}")
                plt.savefig(f"cm_w_{epoch}.png")
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


if __name__ == "__main__":
    main()
