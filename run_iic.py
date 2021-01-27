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
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from umap import UMAP
from sklearn.metrics import silhouette_samples
from matplotlib.lines import Line2D

from src.utils.functional import (
    acronym,
    darken,
    cmap_with_marker,
    pca,
    cosine_similarity,
    compute_serial_matrix,
    sample_from_each_class,
    segmented_cmap,
    modify_cax,
)
from src.utils import transforms
from src.data import datasets
from src.data import samplers
from src import config

from mnist import IIC

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use("seaborn-poster")
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
plt.rcParams["lines.markersize"] = 6.0
plt.rcParams["text.usetex"] = True
plt.rc("legend", fontsize=10)


@hydra.main(config_path="config", config_name="test")
def main(args):
    # random
    random.seed(args.seed)
    # Numpy
    np.random.seed(args.seed)
    # Pytorch
    torch.manual_seed(args.seed)

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

    targets = np.array([acronym(target) for target in args.targets])

    dataset = datasets.HDF5(args.dataset_root, transform_fn, target_transform_fn)
    train_set, test_set = copy.copy(dataset), dataset.sample(args.num_test_samples, stratify=dataset.targets)
    train_set = datasets.co(train_set, transform_fn, augment_fn)

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

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    model = IIC(
        ch_in=args.ch_in,
        dim_w=args.dim_w,
        dim_w_over=args.dim_w_over,
        dim_z=args.dim_z,
        num_heads=args.num_heads,
    ).to(device)
    if args.load_state_dict:
        model.load_state_dict_part(torch.load(os.path.join(args.model_dir, "model_m1_usl.pt")))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)

    stats = defaultdict(lambda: [])

    for epoch in range(args.num_epochs):
        print(f"----- training at epoch {epoch} -----")
        model.train()
        total = 0
        total_dict = defaultdict(lambda: 0)
        for (x, x_), _ in tqdm(train_loader):
            x = x.to(device)
            x_ = x_.to(device)
            mi, mi_over = model(x, x_, lam=args.lam, detach=args.z_detach)
            loss = sum([mi, mi_over])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            total_dict["total loss"] += loss.item()
            total_dict["mutual information loss"] += mi.item()
            total_dict["mutual information loss for overclustering"] += mi_over.item()
        for key, value in total_dict.items():
            print("{}: {:.3f} at epoch: {}".format(key, value, epoch))
            stats[key].append(value)

        # scheduler.step()

        if epoch % args.eval_interval == 0:
            print(f"----- evaluating at epoch {epoch} -----")
            model.eval()
            params = defaultdict(lambda: [])
            num_samples = 0
            with torch.no_grad():
                for i, (x, y) in tqdm(enumerate(test_loader)):
                    x = x.to(device)
                    qz, pi, pi_over = model.get_params(x)
                    pred = torch.argmax(pi, 1)
                    pred_over = torch.argmax(pi_over, 1)

                    params["y"].append(y)
                    params["pred"].append(pred.cpu())
                    params["pred_over"].append(pred_over.cpu())
                    params["pi"].append(pi.cpu())
                    params["qz"].append(qz.cpu())
                    num_samples += x.shape[0]

                y = torch.cat(params["y"]).numpy().astype(int)
                pred = torch.cat(params["pred"]).numpy().astype(int)
                pred_over = torch.cat(params["pred_over"]).numpy().astype(int)
                pi = torch.cat(params["pi"]).numpy().astype(float)
                qz = torch.cat(params["qz"]).numpy().astype(float)

                if args.num_heads > 1:
                    hg = torch.cat(params["pi"]).view(num_samples, -1).numpy().astype(float)
                    try:
                        hg = PCA(n_components=64).fit_transform(hg)
                    except:
                        pass
                    print("Computing cosine similarity matrix...")
                    simmat = cosine_similarity(torch.from_numpy(hg))
                    print("Computing cosine distance reordered matrix...")
                    dist_mat, reordered, _ = compute_serial_matrix(simmat)
                    pred_ensembled = SpectralClustering(n_clusters=args.dim_w, random_state=args.seed).fit(simmat).labels_
                    simmat_reordered = simmat[reordered][:, reordered]
                else:
                    simmat = cosine_similarity(torch.from_numpy(qz))
                    dist_mat, reordered, _ = compute_serial_matrix(simmat)
                    simmat_reordered = simmat[reordered][:, reordered]

                figure = plt.figure()
                grid = ImageGrid(figure, 111, nrows_ncols=(2, 1), axes_pad=0.05)
                im0 = grid[0].imshow(simmat_reordered, aspect=1)
                grid[0].set_xticklabels([])
                grid[0].set_yticklabels([])
                grid[0].set_ylabel("cosine similarity")
                axins0 = inset_axes(
                    grid[0],
                    height=0.2,
                    width="100%",
                    loc="upper left",
                    bbox_to_anchor=(0, 0.05, 1, 1),
                    bbox_transform=grid[0].transAxes,
                    borderpad=0,
                )
                im0.set_clim(0, 1)
                cb0 = plt.colorbar(im0, cax=axins0, orientation="horizontal")
                cb0.set_ticks(np.linspace(-1, 1, 5))
                axins0.xaxis.set_ticks_position("top")

                im1 = grid[1].imshow(y[reordered][np.newaxis, :], aspect=100, cmap=segmented_cmap(len(targets), "tab20b"))
                grid[1].set_xticklabels([])
                grid[1].set_yticklabels([])
                grid[1].set_ylabel("label")
                axins1 = inset_axes(
                    grid[1],
                    height=0.2,
                    width="100%",
                    loc="lower left",
                    bbox_to_anchor=(0, -0.95, 1, 1),
                    bbox_transform=grid[1].transAxes,
                    borderpad=0,
                )
                im1.set_clim(0 - 0.5, len(targets) - 0.5)
                cb1 = plt.colorbar(im1, cax=axins1, orientation="horizontal")
                cb1.set_ticks(np.arange(0, len(targets), 2))
                cb1.set_ticklabels(targets[np.arange(0, len(targets), 2)])
                plt.setp(axins1.get_xticklabels(), rotation=45, horizontalalignment="right")
                axins1.xaxis.set_ticks_position("bottom")
                plt.savefig(f"simmat_e{epoch}.png", bbox_inches="tight", dpi=300)
                plt.close()

                if epoch > 0:
                    for key, value in stats.items():
                        plt.plot(value)
                        plt.ylabel(key)
                        plt.xlabel("epoch")
                        plt.title(key)
                        plt.xlim((0, len(value) - 1))
                        fbase = key.replace(" ", "_")
                        plt.tight_layout()
                        plt.savefig(f"{fbase}_e{epoch}.png", dpi=300)
                        plt.close()

                if args.num_heads > 1:
                    print(f"Plotting confusion matrix with ensembled label as head {i}...")
                    fig, ax = plt.subplots()
                    cm = confusion_matrix(y, pred_ensembled, labels=list(range(args.dim_w)))
                    cm = cm[: args.num_classes, :]
                    cmn = normalize(cm, axis=0)
                    sns.heatmap(
                        cmn,
                        ax=ax,
                        linewidths=0.1,
                        linecolor="gray",
                        cmap="Greens",
                        cbar=True,
                        yticklabels=targets,
                        cbar_kws={"aspect": 50, "pad": 0.01, "anchor": (0, 0.05)},
                    )
                    plt.yticks(rotation=45)
                    plt.xlabel("ensemble predicted labels")
                    plt.ylabel("true labels")
                    ax.set_title(r"confusion matrix at epoch %d with ensemble classifier" % (epoch))
                    plt.tight_layout()
                    plt.savefig(f"cm_ensembled_e{epoch}.png", dpi=300)
                    plt.close()

                for i in range(args.num_heads):
                    try:
                        pred_i, pred_over_i = pred[..., i], pred_over[..., i]
                    except:
                        pred_i, pred_over_i = pred, pred_over
                    print(f"Plotting confusion matrix with predicted label as head {i}...")
                    fig, ax = plt.subplots()
                    cm = confusion_matrix(y, pred_i, labels=list(range(args.dim_w)))
                    cm = cm[: args.num_classes, :]
                    cmn = normalize(cm, axis=0)
                    sns.heatmap(
                        cmn,
                        ax=ax,
                        linewidths=0.1,
                        linecolor="gray",
                        cmap="Greens",
                        cbar=True,
                        yticklabels=targets,
                        cbar_kws={"aspect": 50, "pad": 0.01, "anchor": (0, 0.05)},
                    )
                    plt.yticks(rotation=45)
                    plt.xlabel("predicted labels")
                    plt.ylabel("true labels")
                    ax.set_title(r"confusion matrix at epoch %d with classifier %d" % (epoch, i))
                    plt.tight_layout()
                    plt.savefig(f"cm_c{i}_e{epoch}.png", dpi=300)
                    plt.close()

                    print(f"Plotting confusion matrix with predicted label for overclustering as head {i}...")
                    fig, ax = plt.subplots()
                    cm = confusion_matrix(y, pred_over_i, labels=list(range(args.dim_w_over)))
                    cm = cm[: args.num_classes, :]
                    cmn = normalize(cm, axis=0)
                    sns.heatmap(
                        cmn,
                        ax=ax,
                        linewidths=0.1,
                        linecolor="gray",
                        cmap="Greens",
                        cbar=True,
                        yticklabels=targets,
                        cbar_kws={"aspect": 50, "pad": 0.01, "anchor": (0, 0.05)},
                    )
                    plt.yticks(rotation=45)
                    plt.xlabel("predicted labels (overclustering)")
                    plt.ylabel("true labels")
                    ax.set_title(r"confusion matrix for overclustering at epoch %d with classifier %d" % (epoch, i))
                    plt.tight_layout()
                    plt.savefig(f"cm_over_c{i}_e{epoch}.png", dpi=300)
                    plt.close()

                print("t-SNE decomposing...")
                qz_tsne = TSNE(n_components=2, random_state=args.seed).fit(qz).embedding_

                print(f"Plotting t-SNE 2D latent features with true labels...")
                fig, ax = plt.subplots()
                cmap = segmented_cmap(args.num_classes, "tab20b")
                for i in range(args.num_classes):
                    idx = np.where(y == i)[0]
                    if len(idx) > 0:
                        c = cmap(i)
                        ax.scatter(qz_tsne[idx, 0], qz_tsne[idx, 1], color=c, label=targets[i], edgecolors=darken(c))
                ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", ncol=len(targets) // 25 + 1)
                ax.set_title(r"t-SNE 2D plot of $q(\bm{z})$ with true labels at epoch %d" % (epoch))
                ax.set_aspect(1.0 / ax.get_data_ratio())
                plt.tight_layout()
                plt.savefig(f"qz_tsne_true_e{epoch}.png", dpi=300)
                plt.close()

                # print(f"Plotting t-SNE 2D latent features with pred labels...")
                # fig, ax = plt.subplots()
                # cmap = segmented_cmap(len(np.unique(pred_ensembled)), "tab20b")
                # for i, l in enumerate(np.unique(pred_ensembled)):
                #     idx = np.where(pred == l)[0]
                #     if len(idx) > 0:
                #         c = cmap(i)
                #         ax.scatter(qz_tsne[idx, 0], qz_tsne[idx, 1], color=c, label=l, edgecolors=darken(c))
                # ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", ncol=len(np.unique(pred_ensembled)) // 25 + 1)
                # ax.set_title(r"t-SNE 2D plot  of latent codes with ensemble predicted labels at epoch %d" % (epoch))
                # ax.set_aspect(1.0 / ax.get_data_ratio())
                # plt.tight_layout()
                # plt.savefig(f"qz_tsne_pred_e{epoch}.png")
                # plt.close()

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model_iic.pt"))


if __name__ == "__main__":
    main()
