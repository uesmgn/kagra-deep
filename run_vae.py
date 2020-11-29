import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import abc, defaultdict
import numbers
from tqdm import tqdm
from sklearn.manifold import TSNE

from src.utils import transforms
from src.data import datasets
from src.data import samplers
from src import config

from mnist import M2

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
    model = M2(dim_y=args.num_classes, dim_z=args.dim_z).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = args.weights

    for epoch in range(args.num_epochs):
        print(f"----- training at epoch {epoch} -----")
        model.train()
        total = 0
        total_dict = defaultdict(lambda: 0)
        for i, (x, _) in tqdm(enumerate(train_loader)):
            x = x.to(device)
            loss = model(x, weights=weights)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.total.item()
            for k, v in loss.items():
                total_dict[k] += v.item()
        print("loss: {:.3f} at epoch: {}".format(total, epoch))
        for k, v in total_dict.items():
            print("loss_{}: {:.3f} at epoch: {}".format(k, v, epoch))

        if epoch % args.eval_interval == 0:
            print(f"----- evaluating at epoch {epoch} -----")
            model.eval()
            params = defaultdict(lambda: torch.tensor([]))

            with torch.no_grad():
                for i, (x, y) in tqdm(enumerate(test_loader)):
                    x = x.to(device)
                    qy, qy_pi = model.qy_x(x, hard=True)
                    _, qz, _ = model.qz_xy(x, qy)
                    y_pred = torch.argmax(qy_pi, -1)
                    # y_onehot = F.one_hot(y, num_classes=args.num_classes).float().to(device)
                    # pz, _, _ = model.pz_y(y_onehot.float().to(device))

                    params["qz"] = torch.cat([params["qz"], qz.cpu()])
                    params["y"] = torch.cat([params["y"], y])
                    params["y_pred"] = torch.cat([params["y_pred"], y_pred.cpu()])

                for i in range(args.num_classes):
                    y_i = F.one_hot(torch.full((100,), i).long(), num_classes=args.num_classes)
                    pz, _, _ = model.pz_y(y_i.float().to(device))
                    params["pz"] = torch.cat([params["pz"], pz.cpu()])

                qz = params["qz"].numpy()
                pz = params["pz"].numpy()
                qz = TSNE(n_components=2).fit_transform(qz)
                pz = TSNE(n_components=2).fit_transform(pz)

                y = params["y"].numpy().astype(int)
                y_pred = params["y_pred"].numpy().astype(int)

                plt.figure(figsize=(12, 12))
                for i in np.unique(y):
                    idx = np.where(y == i)
                    plt.scatter(qz[idx, 0], qz[idx, 1], label=i)
                plt.legend()
                plt.title(f"qz_true_{epoch}")
                plt.savefig(f"qz_true_{epoch}.png")
                plt.close()

                plt.figure(figsize=(12, 12))
                for i in np.unique(y_pred):
                    idx = np.where(y_pred == i)
                    plt.scatter(qz[idx, 0], qz[idx, 1], label=i)
                plt.legend()
                plt.title(f"qz_pred_{epoch}")
                plt.savefig(f"qz_pred_{epoch}.png")
                plt.close()

                yy = torch.tensor(list(range(args.num_classes))).unsqueeze(1)
                yy = yy.repeat(1, 100).flatten()

                plt.figure(figsize=(12, 12))
                for i in range(args.num_classes):
                    idx = np.where(yy == i)
                    plt.scatter(pz[idx, 0], pz[idx, 1], label=i)
                plt.legend()
                plt.title(f"pz_{epoch}")
                plt.savefig(f"pz_{epoch}.png")
                plt.close()


if __name__ == "__main__":
    main()
