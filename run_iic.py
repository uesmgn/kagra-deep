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
from sklearn.manifold import TSNE

from src.utils.functional import acronym, darken, colormap, pca, cosine_similarity
from src.utils import transforms
from src.data import datasets
from src.data import samplers
from src import config

from mnist import IIC

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import seaborn as sns
from torchcluster.zoo.spectrum import SpectrumClustering


plt.style.use("seaborn-poster")
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
plt.rc("legend", fontsize=10)
plt.rc("axes", titlesize=10)


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
    train_set, test_set = dataset.split(train_size=args.train_size, stratify=dataset.targets)
    sample_indices = random.sample(range(len(test_set)), 5)
    train_set.transform = augment_fn
    # train_sampler = samplers.Balancer(train_set, args.batch_size * args.num_train_steps)
    train_sampler = samplers.Upsampler(train_set, args.batch_size * args.num_train_steps)

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
    model = IIC(
        ch_in=args.ch_in,
        dim_y=args.num_classes,
        dim_w=args.dim_w,
        dim_z=args.dim_z,
        use_multi_heads=args.use_multi_heads,
        num_heads=args.num_heads,
    ).to(device)

    if args.load_state_dict:
        model.load_state_dict_part(torch.load(os.path.join(args.model_dir, "model.pt")))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=2, T_mult=2)
    stats = defaultdict(lambda: [])

    for epoch in range(args.num_epochs):
        print(f"----- training at epoch {epoch} -----")
        model.train()
        total = 0
        total_dict = defaultdict(lambda: 0)
        for i, (x, y) in tqdm(enumerate(train_loader)):
            x = x.to(device)
            mi_x, mi_v = model(x, z_detach=args.iic_detach, lam=args.lam)
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
                    params["y_pi"].append(y_pi.cpu())
                    params["w_pi"].append(w_pi.cpu())
                    params["y_proba"].append(y_proba.cpu())
                    params["w_proba"].append(w_proba.cpu())
                    num_samples += x.shape[0]

            y = torch.cat(params["y"]).numpy().astype(int)
            y_pred = torch.cat(params["y_pred"]).numpy().astype(int)
            w_pred = torch.cat(params["w_pred"]).numpy().astype(int)
            y_hyp = torch.cat(params["y_pi"]).view(num_samples, -1)
            w_hyp = torch.cat(params["w_pi"]).view(num_samples, -1)
            y_proba = torch.stack(params["y_proba"], -1).sum(-1).numpy()
            w_proba = torch.stack(params["w_proba"], -1).sum(-1).numpy()

            plt.imshow(y_proba)
            plt.savefig(f"y_proba_e{epoch}.png")
            plt.close()

            plt.imshow(w_proba)
            plt.savefig(f"w_proba_e{epoch}.png")
            plt.close()

            if epoch > 0:
                plt.rcParams["text.usetex"] = False
                for key, value in stats.items():
                    plt.figure()
                    plt.plot(value)
                    plt.ylabel(key)
                    plt.xlabel("epoch")
                    plt.title("loss %s" % key)
                    plt.xlim((0, len(value) - 1))
                    plt.tight_layout()
                    plt.savefig(f"loss_{key}_e{epoch}.png")
                    plt.close()

            plt.rcParams["text.usetex"] = True

            w_simmat = cosine_similarity(w_hyp)
            fig = plt.figure(dpi=500)
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
            fig.suptitle("Similar glitch")
            plt.tight_layout()
            plt.savefig(f"simrank_e{epoch}.png")
            plt.close()

            # y_hyp = torch.mm(y_hyp, y_hyp.transpose(0, 1))
            # y_hyp = pca(y_hyp, 6)
            y_pred_ens, _ = SpectrumClustering(args.num_pred_classes, k=32)(w_hyp)

            plt.figure()
            cm = confusion_matrix(y, y_pred_ens, labels=np.arange(args.num_pred_classes))
            cm = cm[: args.num_classes, :]
            cmn = normalize(cm, axis=0)
            sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", cbar=False, yticklabels=targets)
            plt.yticks(rotation=45)
            plt.title(r"confusion matrix $\bm{y}$ with $q(\bm{y})$ ensembled at epoch %d" % epoch)
            plt.tight_layout()
            plt.savefig(f"cm_y_qz_ensembled_e{epoch}.png")
            plt.close()

            for j in range(0, args.num_heads, 3):
                plt.figure()
                cm = confusion_matrix(y, y_pred[:, j], labels=np.arange(args.num_classes))
                cm = cm[: args.num_classes, :]
                cmn = normalize(cm, axis=0)
                sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", cbar=False, yticklabels=targets)
                plt.yticks(rotation=45)
                plt.title(r"confusion matrix $\bm{y}$ with $q(\bm{y})$ by head %d at epoch %d" % (j, epoch))
                plt.tight_layout()
                plt.savefig(f"cm_y_h{j}_e{epoch}.png")
                plt.close()

            if epoch % args.embedding_interval == 0:

                # latent features
                qz = torch.cat(params["qz"]).numpy()
                mapper = TSNE(n_components=2, random_state=args.seed)
                qz = mapper.fit_transform(qz)
                # mapper = umap.UMAP(random_state=args.seed).fit(qz)
                # qz = mapper.embedding_

                plt.figure()
                for i in range(args.num_classes):
                    idx = np.where(y == i)[0]
                    if len(idx) > 0:
                        c = colormap(i)
                        plt.scatter(qz[idx, 0], qz[idx, 1], c=c, label=targets[i], edgecolors=darken(c))
                plt.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
                plt.title(r"$q(\bm{z})$ at epoch %d" % (epoch))
                plt.tight_layout()
                plt.savefig(f"qz_true_e{epoch}.png")
                plt.close()

                plt.figure()
                for i in range(args.num_pred_classes):
                    idx = np.where(y_pred_ens == i)[0]
                    if len(idx) > 0:
                        c = colormap(i)
                        plt.scatter(qz[idx, 0], qz[idx, 1], c=c, label=i, edgecolors=darken(c))
                plt.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
                plt.title(r"$q(\bm{z})$ ensembled at epoch %d" % (epoch))
                plt.tight_layout()
                plt.savefig(f"qz_ensembled_e{epoch}.png")
                plt.close()

                # plt.figure()
                # cm = confusion_matrix(y, w_pred[:, j], labels=np.arange(args.dim_w))
                # cm = cm[: args.num_classes, :]
                # cmn = normalize(cm, axis=0) * normalize(cm, axis=1)
                # sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", cbar=False, yticklabels=targets)
                # plt.yticks(rotation=45)
                # plt.title(r"confusion matrix $\bm{y}$ with $q(\bm{w})$ by head %d at epoch %d" % (j, epoch))
                # plt.tight_layout()
                # plt.savefig(f"cm_w_h{j}_e{epoch}.png")
                # plt.close()

                for j in range(0, args.num_heads, 3):
                    plt.figure()
                    for i in np.unique(y):
                        idx = np.where(y_pred[:, j] == i)[0]
                        if len(idx) > 0:
                            c = colormap(i)
                            plt.scatter(qz[idx, 0], qz[idx, 1], c=c, label=i, edgecolors=darken(c))
                    plt.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
                    plt.title(r"$q(\bm{z})$ labeled by head %d at epoch %d" % (j, epoch))
                    plt.tight_layout()
                    plt.savefig(f"qz_h{j}_e{epoch}.png")
                    plt.close()

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model_iic.pt"))


if __name__ == "__main__":
    main()
