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
from umap import UMAP
from sklearn.metrics import silhouette_samples

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

from mnist import VAE

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
    model = VAE(ch_in=args.ch_in, dim_z=args.dim_z).to(device)
    if args.load_state_dict:
        model.load_state_dict_part(torch.load(args.model_path))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)

    stats = defaultdict(lambda: [])

    for epoch in range(args.num_epochs):
        print(f"----- training at epoch {epoch} -----")
        model.train()
        total = 0
        total_dict = defaultdict(lambda: 0)
        for x, _ in tqdm(train_loader):
            x = x.to(device)
            bce, kl_gauss = model(x)
            loss = sum([bce, kl_gauss])
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

        # scheduler.step()

        if epoch % args.eval_interval == 0:
            print(f"----- evaluating at epoch {epoch} -----")
            model.eval()
            params = defaultdict(lambda: torch.tensor([]))

            with torch.no_grad():
                for i, (x, y) in tqdm(enumerate(test_loader)):
                    x = x.to(device)
                    _, qz, _ = model.qz_x(x)

                    params["y"] = torch.cat([params["y"], y])
                    params["qz"] = torch.cat([params["qz"], qz.cpu()])

                y = params["y"].numpy().astype(int)
                qz = params["qz"].numpy()

                plt.rcParams["text.usetex"] = False
                if epoch > 0:
                    for key, value in stats.items():
                        plt.plot(value)
                        plt.ylabel(key)
                        plt.xlabel("epoch")
                        plt.title(key)
                        plt.xlim((0, len(value) - 1))
                        plt.savefig(f"loss_{key}_e{epoch}.png", dpi=args.dpi)
                        plt.close()

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model_m1_usl.pt"))

        plt.rcParams["text.usetex"] = True

        if epoch % args.embedding_interval == 0 and epoch > 0:
            print("t-SNE decomposing...")
            qz_tsne = TSNE(n_components=2, random_state=args.seed).fit(qz).embedding_
            print("UMAP decomposing...")
            qz_umap = UMAP(n_components=2, min_dist=0.2, random_state=args.seed).fit(qz).embedding_

            print(f"Plotting 2D latent features with true labels...")
            fig, ax = plt.subplots()
            cmap = segmented_cmap(args.num_classes, "tab20b")
            for i in range(args.num_classes):
                idx = np.where(y == i)[0]
                if len(idx) > 0:
                    c = cmap(i)
                    ax.scatter(qz_tsne[idx, 0], qz_tsne[idx, 1], color=c, label=targets[i], edgecolors=darken(c))
            ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
            ax.set_title(r"2d $q(\bm{z})$ at epoch %d (t-SNE)" % (epoch))
            ax.set_aspect(1.0 / ax.get_data_ratio())
            plt.tight_layout()
            plt.savefig(f"qz_true_e{epoch}.png", dpi=args.dpi)
            plt.close()

            fig, ax = plt.subplots()
            cmap = segmented_cmap(args.num_classes, "tab20b")
            for i in range(args.num_classes):
                idx = np.where(y == i)[0]
                if len(idx) > 0:
                    c = cmap(i)
                    ax.scatter(qz_umap[idx, 0], qz_umap[idx, 1], color=c, label=targets[i], edgecolors=darken(c))
            ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
            ax.set_title(r"2d $q(\bm{z})$ at epoch %d (UMAP)" % (epoch))
            ax.set_aspect(1.0 / ax.get_data_ratio())
            plt.tight_layout()
            plt.savefig(f"qz_umap_e{epoch}.png", dpi=args.dpi)
            plt.close()

            sample_silhouette_values = silhouette_samples(qz, y)
            y_lower = 10
            cmap = segmented_cmap(len(args.targets), "tab20b")
            fig, ax = plt.subplots()
            for i in np.unique(y):
                ith_cluster_silhouette_values = sample_silhouette_values[y == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                c = cmap(i)
                ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=c, edgecolor=c, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax.text(-0.1, y_lower + 0.5 * size_cluster_i, targets[i], color=c)

                # Compute the new y_lower for next plot
                y_lower = y_upper + 50  # 10 for the 0 samples

                ax.set_title("Silhouette coefficient for each cluster")
                ax.set_xlabel("silhouette coefficient")
                ax.set_ylabel("label")

                ax.set_yticks([])  # Clear the yaxis labels / ticks

            plt.tight_layout()
            plt.savefig(f"silhouette_e{epoch}.png", dpi=args.dpi)
            plt.close()


if __name__ == "__main__":
    main()
