import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import abc, defaultdict
import numbers
from tqdm import tqdm
import copy
import umap
from itertools import cycle
import os
from sklearn.manifold import TSNE

from src.utils.functional import (
    acronym,
    darken,
    cmap_with_marker,
    pca,
    cosine_similarity,
    compute_serial_matrix,
    sample_from_each_class,
    segmented_cmap,
)
from src.utils import transforms
from src.data import datasets
from src.data import samplers
from src import config

from mnist import CVAE

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.style.use("seaborn-poster")
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
plt.rc("legend", fontsize=10)
plt.rc("axes", titlesize=10)
plt.rcParams["lines.markersize"] = 5.0


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
    train_set, test_set = copy.copy(dataset), dataset.sample(args.num_test_samples)
    # train_set, test_set = dataset.split(train_size=args.train_size, stratify=dataset.targets)
    train_set.transform = augment_fn

    def sampler_callback(ds, num_samples):
        return samplers.Upsampler(ds, num_samples)

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
    model = CVAE(ch_in=args.ch_in, dim_y=100, dim_z=512).to(device)
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
            bce, kl_gauss, kl_cat, mi = model(x)
            loss = sum([bce, kl_gauss, kl_cat, mi])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            total_dict["total"] += loss.item()
            total_dict["bce"] += bce.item()
            total_dict["kl_gauss"] += kl_gauss.item()
            total_dict["kl_cat"] += kl_cat.item()
            total_dict["mi"] += mi.item()
        for key, value in total_dict.items():
            print("loss_{}: {:.3f} at epoch: {}".format(key, value, epoch))
            stats[key].append(value)

        scheduler.step()

        if epoch % args.eval_interval == 0:
            print(f"----- evaluating at epoch {epoch} -----")
            model.eval()
            params = defaultdict(list)

            with torch.no_grad():
                for i, (x, y) in tqdm(enumerate(test_loader)):
                    x = x.to(device)
                    qz, pi = model.get_params(x)
                    pred = torch.argmax(pi, -1)

                    params["y"].append(y)
                    params["pred"].append(pred.cpu())
                    params["qz"].append(qz.cpu())

                y = torch.cat(params["y"]).numpy().astype(int)
                pred = torch.cat(params["pred"]).numpy().astype(int)
                qz = torch.cat(params["qz"]).numpy()

                print(f"Computing 2D latent features by t-SNE...")
                # latent features
                qz = TSNE(n_components=2, metric="cosine", random_state=args.seed).fit(qz).embedding_

                plt.rcParams["text.usetex"] = False
                if epoch > 0:
                    for key, value in stats.items():
                        plt.plot(value)
                        plt.ylabel(key)
                        plt.xlabel("epoch")
                        plt.title(key)
                        plt.xlim((0, len(value) - 1))
                        plt.savefig(f"loss_{key}_e{epoch}.png")
                        plt.close()

                plt.rcParams["text.usetex"] = True

                print(f"Plotting 2D latent features with true labels...")
                fig, ax = plt.subplots()
                cmap = segmented_cmap(args.num_classes, "tab20b")
                for i in range(args.num_classes):
                    idx = np.where(y == i)[0]
                    if len(idx) > 0:
                        c = cmap(i)
                        ax.scatter(qz[idx, 0], qz[idx, 1], color=c, label=targets[i], edgecolors=darken(c))
                ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
                ax.set_title(r"$q(\bm{z})$ with true labels at epoch %d" % (epoch))
                ax.set_aspect(1.0 / ax.get_data_ratio())
                plt.tight_layout()
                plt.savefig(f"qz_true_e{epoch}.png", transparent=True, dpi=args.dpi)
                plt.close()

                print(f"Plotting 2D latent features with pred labels...")
                fig, ax = plt.subplots()
                cmap = segmented_cmap(len(np.unique(pred)), "tab20b")
                for i in np.unique(pred):
                    idx = np.where(pred == i)[0]
                    if len(idx) > 0:
                        c = cmap(i)
                        ax.scatter(qz[idx, 0], qz[idx, 1], color=c, label=i, edgecolors=darken(c))
                ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", ncol=np.ceil(len(np.unique(pred)) / 20).astype(int))
                ax.set_title(r"$q(\bm{z})$ with pred labels at epoch %d" % (epoch))
                ax.set_aspect(1.0 / ax.get_data_ratio())
                plt.tight_layout()
                plt.savefig(f"qz_pred_e{epoch}.png", transparent=True, dpi=args.dpi)
                plt.close()

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model_m1_usl.pt"))


if __name__ == "__main__":
    main()
