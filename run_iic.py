import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import abc, defaultdict
import numbers
from tqdm import tqdm
import umap
from sklearn.manifold import TSNE

from src.utils.functional import acronym, darken, colormap
from src.utils import transforms
from src.data import datasets
from src.data import samplers
from src import config

from mnist import IIC

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.style.use("seaborn-poster")
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


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

    targets = np.array([acronym(target) for target in args.targets])

    dataset = datasets.HDF5(args.dataset_root, transform_fn, target_transform_fn)
    train_set, test_set = dataset.split(train_size=args.train_size, stratify=dataset.targets)
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
        model.load_state_dict_part(torch.load(args.model_path))

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=2, T_mult=2)
    stats = defaultdict(lambda: [])

    for epoch in range(args.num_epochs):
        print(f"----- training at epoch {epoch} -----")
        model.train()
        total = 0
        total_dict = defaultdict(lambda: 0)
        for i, (x, _) in tqdm(enumerate(train_loader)):
            x = x.to(device)
            mi_x, mi_v = model(x, args.iic_detach)
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
            params = defaultdict(lambda: torch.tensor([]))
            indices = torch.tensor([]).long()
            with torch.no_grad():
                for i, (x, y) in tqdm(enumerate(test_loader)):
                    x = x.to(device)
                    qz = model.embedding(x)
                    y_pi, w_pi = model.clustering(qz)
                    y_pred = torch.argmax(y_pi, -1)
                    w_pred = torch.argmax(w_pi, -1)

                    params["y"] = torch.cat([params["y"], y])
                    params["y_pred"] = torch.cat([params["y_pred"], y_pred.cpu()])
                    params["w_pred"] = torch.cat([params["w_pred"], w_pred.cpu()])
                    params["qz"] = torch.cat([params["qz"], qz.cpu()])

                y = params["y"].numpy().astype(int)
                y_pred = params["y_pred"].numpy().astype(int)
                w_pred = params["w_pred"].numpy().astype(int)
                qz = params["qz"].numpy()
                umapper = umap.UMAP(random_state=123).fit(qz)
                qz = umapper.embedding_

                for j in range(args.num_heads):
                    plt.figure()
                    cm = confusion_matrix(y, y_pred[:, j], labels=np.arange(args.num_classes))
                    cm = cm[: args.num_classes, :]
                    cm = (cm - np.mean(cm, axis=1)) / np.std(cm, axis=1)
                    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, yticklabels=targets)
                    plt.yticks(rotation=45)
                    plt.title(r"confusion matrix $\boldsymbol{{y}}$ with $q(\boldsymbol{{y}})$ by head {} at epoch {}".format(j, epoch))
                    plt.tight_layout()
                    plt.savefig(f"cm_y_h{j}_e{epoch}.png")
                    plt.close()

                    plt.figure()
                    cm = confusion_matrix(y, w_pred[:, j], labels=np.arange(args.dim_w))
                    cm = cm[: args.num_classes, :]
                    cm = (cm - np.mean(cm, axis=1)) / np.std(cm, axis=1)
                    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, yticklabels=targets)
                    plt.yticks(rotation=45)
                    plt.title(r"confusion matrix $\boldsymbol{{y}}$ with $q(\boldsymbol{{w}})$ by head {} at epoch {}".format(j, epoch))
                    plt.tight_layout()
                    plt.savefig(f"cm_w_h{j}_e{epoch}.png")
                    plt.close()

                plt.figure()
                for i in np.unique(y):
                    idx = np.where(y == i)[0]
                    if len(idx) > 0:
                        c = colormap(i)
                        plt.scatter(qz[idx, 0], qz[idx, 1], c=c, label=targets[i], edgecolors=darken(c))
                plt.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
                plt.title(r"$q(\boldsymbol{{z}})$ at epoch {}".format(epoch))
                plt.tight_layout()
                plt.savefig(f"qz_true_e{epoch}.png")
                plt.close()

                for j in range(args.num_heads):
                    plt.figure()
                    for i in np.unique(y):
                        idx = np.where(y_pred[:, j] == i)[0]
                        if len(idx) > 0:
                            c = colormap(i)
                            plt.scatter(qz[idx, 0], qz[idx, 1], c=c, label=i, edgecolors=darken(c))
                    plt.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
                    plt.title(r"$q(\boldsymbol{{z}})$ labeled by head {} at epoch {}".format(j, epoch))
                    plt.tight_layout()
                    plt.savefig(f"qz_true_h{j}_e{epoch}.png")
                    plt.close()

                if epoch > 0:
                    for key, value in stats.items():
                        plt.figure()
                        plt.plot(value)
                        plt.ylabel(key)
                        plt.xlabel("epoch")
                        plt.title(key)
                        plt.xlim((0, len(value) - 1))
                        plt.tight_layout()
                        plt.savefig(f"loss_{key}_e{epoch}.png")
                        plt.close()


if __name__ == "__main__":
    main()
