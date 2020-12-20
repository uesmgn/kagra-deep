import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import abc, defaultdict
import numbers
from tqdm import tqdm
import umap
import random
import os
import copy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import cluster as cl
import scipy.linalg

from src.utils.functional import acronym, darken, cmap_with_marker, pca, cosine_similarity, compute_serial_matrix, sample_from_each_class
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


plt.style.use("dark_background")
plt.style.use("seaborn-poster")
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
plt.rc("legend", fontsize=10)
plt.rc("axes", titlesize=10)
plt.rcParams["lines.markersize"] = 6.0


@hydra.main(config_path="config", config_name="test")
def main(args):

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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
    train_set, test_set = copy.copy(dataset), dataset.sample(args.num_test_samples)
    sample_indices = random.sample(range(len(test_set)), 5)
    # train_set, test_set = dataset.split(train_size=args.train_size, stratify=dataset.targets)
    train_set.transform = augment_fn
    # train_sampler = samplers.Balancer(train_set, args.batch_size * args.num_train_steps)
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
    model = IIC(
        ch_in=args.ch_in,
        dim_y=args.num_classes,
        dim_w=args.dim_w,
        dim_z=args.dim_z,
        use_multi_heads=args.use_multi_heads,
        num_heads=args.num_heads,
    ).to(device)

    if args.load_state_dict:
        model.load_state_dict_part(torch.load(os.path.join(args.model_dir, "model_m1_usl.pt")))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)
    stats = defaultdict(lambda: [])
    sc = cl.SpectralClustering(n_clusters=args.num_pred_classes, affinity="nearest_neighbors", random_state=args.seed)

    for epoch in range(args.num_epochs):
        print(f"----- training at epoch {epoch} -----")
        model.train()
        total = 0
        total_dict = defaultdict(lambda: 0)
        for i, (x, y) in tqdm(enumerate(train_loader)):
            x = x.to(device)
            mi_x, mi_v = model(x, z_detach=args.z_detach, lam=args.lam)
            loss = mi_x + mi_v
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            total_dict["total"] += loss.item()
            total_dict["mi_x"] += mi_x.item()
            total_dict["mi_v"] += mi_v.item()
        for key, value in total_dict.items():
            print("loss_{}: {:.3f} at epoch: {}".format(key, value, epoch))
            stats[key].append(value)

        scheduler.step()

        if epoch % args.eval_interval == 0:
            print(f"----- evaluating at epoch {epoch} -----")
            model.eval()
            params = defaultdict(list)
            indices = torch.tensor([]).long()
            num_samples = 0
            with torch.no_grad():
                for i, (x, y) in tqdm(enumerate(test_loader)):
                    x = x.to(device)
                    qz, y_pi, w_pi, y_proba, w_proba = model.get_params(x)
                    y_pred = torch.argmax(y_pi, -1)
                    w_pred = torch.argmax(w_pi, -1)

                    params["y"].append(y)
                    params["y_pred"].append(y_pred.cpu())
                    params["w_pred"].append(w_pred.cpu())
                    # params["y_pi"].append(y_pi.cpu())
                    params["w_pi"].append(w_pi.cpu())
                    # params["y_proba"].append(y_proba.cpu())
                    # params["w_proba"].append(w_proba.cpu())
                    params["qz"].append(qz.cpu())
                    num_samples += x.shape[0]

            y = torch.cat(params["y"]).numpy().astype(int)
            y_pred = torch.cat(params["y_pred"]).numpy().astype(int)
            w_pred = torch.cat(params["w_pred"]).numpy().astype(int)
            # y_hyp = torch.cat(params["y_pi"]).view(num_samples, -1)
            w_hyp = torch.cat(params["w_pi"]).view(num_samples, -1)
            # y_proba = torch.stack(params["y_proba"], -1).sum(-1).numpy()
            # w_proba = torch.stack(params["w_proba"], -1).sum(-1).numpy()

            print("Computing cosine similarity matrix...")
            w_simmat = cosine_similarity(w_hyp)
            print("Computing cosine distance reordered matrix...")
            w_simmat_reordered, reordered, _ = compute_serial_matrix(w_simmat)
            # y_hyp = torch.mm(y_hyp, y_hyp.transpose(0, 1))
            # w_hyp = PCA(n_components=64, random_state=args.seed).fit_transform(w_simmat)
            print("Computing eigen values and vectors...")
            eigs, eigv = scipy.linalg.eigh(w_simmat)
            print("Fitting eigen vectors to Spectral Clustering model...")
            y_pred_sc = sc.fit(eigv[:, -64:]).labels_

            print("Sampling from each predicted classes...")
            samples_fec = sample_from_each_class(y_pred_sc, num_samples=args.num_ranking)

            plt.rcParams["text.usetex"] = False

            fig, ax = plt.subplots()
            ax.plot(eigs[::-1])
            ax.set_xlim(0, len(eigs) - 1)
            ax.set_title("eigh values of similarity matrix at epoch %d" % epoch)
            ax.set_xlabel("order")
            ax.set_ylabel("eigen values")
            ax.set_xlim((0, 100 - 1))
            ax.set_yscale("log")
            ax.set_ylim((1e-3, None))
            plt.tight_layout()
            plt.savefig(f"eigen_e{epoch}.png", transparent=True)
            plt.close()

            if epoch > 0:
                for key, value in stats.items():
                    fig, ax = plt.subplots()
                    ax.plot(value)
                    ax.set_ylabel(key)
                    ax.set_xlabel("epoch")
                    ax.set_title("loss %s" % key)
                    ax.set_xlim((0, len(value) - 1))
                    plt.tight_layout()
                    plt.savefig(f"loss_{key}_e{epoch}.png", transparent=True)
                    plt.close()

            fig = plt.figure(dpi=200)
            axs = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0)
            axs[0].imshow(w_simmat_reordered, aspect=1)
            axs[0].axis("off")
            axs[1].imshow(y_pred_sc[reordered][np.newaxis, :], aspect=300)
            axs[1].axis("off")
            fig.suptitle("cosine similarity matrix with SC clusters at epoch %d" % epoch)
            plt.savefig(f"w_simmat_sc_e{epoch}.png", transparent=True)
            plt.close()

            plt.rcParams["text.usetex"] = True

            for i, (label, indices) in enumerate(samples_fec.items()):
                if i % 5 == 0:
                    fig, _ = plt.subplots(dpi=200)
                    print(f"Plotting samples from each predicted classes {i // 5}...")
                for n, m in enumerate(indices):
                    x, _ = test_set[m]
                    ax = plt.subplot(5, args.num_ranking, args.num_ranking * (i % 5) + n + 1)
                    ax.imshow(x[0])
                    ax.axis("off")
                    ax.margins(0)
                    ax.set_title(r"$\bm{x}_{(%d)} \in \bm{y}_{(%d)}$" % (m, label))
                if i % 5 == 4:
                    plt.subplots_adjust(wspace=0.05, top=0.92, bottom=0.05, left=0.05, right=0.95)
                    fig.suptitle("Random samples from each predicted labels")
                    plt.tight_layout()
                    plt.savefig(f"samples_{i // 5}_e{epoch}.png", transparent=True)
                    plt.close()

            print(f"Plotting random samples with 5 most similar samples...")
            fig, _ = plt.subplots(dpi=200)
            for i, j in enumerate(sample_indices):
                x, _ = test_set[j]
                ax = plt.subplot(len(sample_indices), args.num_ranking + 2, (args.num_ranking + 2) * i + 1)
                ax.imshow(x[0])
                ax.axis("off")
                ax.margins(0)
                ax.set_title(r"$\bm{x}_{(%d)}$" % j)
                sim, sim_indices = torch.sort(w_simmat[j, :], descending=True)
                sim, sim_indices = sim[1 : args.num_ranking + 1], sim_indices[1 : args.num_ranking + 1]
                for n, m in enumerate(sim_indices):
                    ax = plt.subplot(len(sample_indices), args.num_ranking + 2, (args.num_ranking + 2) * i + 3 + n)
                    x, _ = test_set[m]
                    ax.imshow(x[0])
                    ax.axis("off")
                    ax.margins(0)
                    ax.set_title(r"%.2f" % sim[n])
            plt.subplots_adjust(wspace=0.05, top=0.92, bottom=0.05, left=0.05, right=0.95)
            fig.suptitle("Random samples with corresponding similar glitches")
            plt.tight_layout()
            plt.savefig(f"simrank_e{epoch}.png", transparent=True)
            plt.close()

            print(f"Plotting confusion matrix with ensembled label...")
            fig, ax = plt.subplots(dpi=200)
            cm = confusion_matrix(y, y_pred_sc)
            cm = cm[: args.num_classes, :]
            cmn = normalize(cm, axis=0)
            sns.heatmap(cmn, ax=ax, annot=cm, fmt="d", linewidths=0.1, cmap="Greens", cbar=False, yticklabels=targets, annot_kws={"fontsize": 8})
            plt.yticks(rotation=45)
            ax.set_title(r"confusion matrix $\bm{y}$ with $q(\bm{y})$ ensembled with SC at epoch %d" % epoch)
            plt.tight_layout()
            plt.savefig(f"cm_sc_e{epoch}.png", transparent=True)
            plt.close()

            # for j in range(0, args.num_heads, 3):
            #     plt.figure(dpi=500)
            #     cm = confusion_matrix(y, y_pred[:, j], labels=np.arange(args.num_classes))
            #     cm = cm[: args.num_classes, :]
            #     cmn = normalize(cm, axis=0)
            #     sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", cbar=False, yticklabels=targets)
            #     plt.yticks(rotation=45)
            #     plt.title(r"confusion matrix $\bm{y}$ with $q(\bm{y})$ by head %d at epoch %d" % (j, epoch))
            #     plt.tight_layout()
            #     plt.savefig(f"cm_y_h{j}_e{epoch}.png", transparent=True)
            #     plt.close()

        if epoch % args.embedding_interval == 0:

            print(f"Computing 2D latent features by t-SNE...")
            # latent features
            qz = torch.cat(params["qz"]).numpy()
            qz = TSNE(n_components=2, metric="cosine", random_state=args.seed).fit(qz).embedding_
            # qz = umap.UMAP(n_components=2, random_state=args.seed).fit(qz).embedding_

            print(f"Plotting 2D latent features with true labels...")
            fig, ax = plt.subplots(dpi=200)
            for i in range(args.num_classes):
                idx = np.where(y == i)[0]
                if len(idx) > 0:
                    c, m = cmap_with_marker(i)
                    ax.scatter(qz[idx, 0], qz[idx, 1], color=c, marker=m, label=targets[i], edgecolors=darken(c))
            ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
            ax.set_title(r"$q(\bm{z})$ at epoch %d" % (epoch))
            ax.set_aspect("equal")
            plt.tight_layout()
            plt.savefig(f"qz_true_e{epoch}.png", transparent=True)
            plt.close()

            for j in range(0, args.num_heads, 3):
                print(f"Plotting 2D latent features with labels by weak classifier-{j}...")
                fig, ax = plt.subplots(dpi=200)
                for i in np.unique(y):
                    idx = np.where(y_pred[:, j] == i)[0]
                    if len(idx) > 0:
                        c, m = cmap_with_marker(i)
                        ax.scatter(qz[idx, 0], qz[idx, 1], color=c, marker=m, label=i, edgecolors=darken(c))
                ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
                ax.set_title(r"$q(\bm{z})$ labeled by head %d at epoch %d" % (j, epoch))
                ax.set_aspect("equal")
                plt.tight_layout()
                plt.savefig(f"qz_h{j}_e{epoch}.png", transparent=True)
                plt.close()

            print(f"Plotting 2D latent features with ensembled labels...")
            fig, ax = plt.subplots(dpi=200)
            for i in range(args.num_pred_classes):
                idx = np.where(y_pred_sc == i)[0]
                if len(idx) > 0:
                    c, m = cmap_with_marker(i)
                    ax.scatter(qz[idx, 0], qz[idx, 1], color=c, marker=m, label=i, edgecolors=darken(c))
            ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", ncol=2)
            ax.set_title(r"$q(\bm{z})$ ensembled at epoch %d" % (epoch))
            ax.set_aspect("equal")
            plt.tight_layout()
            plt.savefig(f"qz_sc_e{epoch}.png", transparent=True)
            plt.close()

        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.model_dir, "model_iic.pt")
            print(f"Saving state dict to {save_path}...")
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
