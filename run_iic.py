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

from mnist import IIC

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


@hydra.main(config_path="config", config_name="test_iic")
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

    dataset = datasets.HDF5(args.dataset_root, transform_fn, target_transform_fn)
    train_set, test_set = dataset.split(train_size=args.train_size, stratify=dataset.targets)
    train_set = datasets.co(train_set, augment_fn)
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
    model = IIC(dim_y=args.num_classes, dim_w=30).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = args.weights

    for epoch in range(args.num_epochs):
        print(f"----- training at epoch {epoch} -----")
        model.train()
        total = 0
        total_dict = defaultdict(lambda: 0)
        for i, (data, _) in tqdm(enumerate(train_loader)):
            x, v = data
            x = x.to(device)
            v = v.to(device)
            loss = model(x, v, weights=weights)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.total.item()
            for key, loss_i in loss.items():
                total_dict[key] += loss_i.item()
        print("loss: {:.3f} at epoch: {}".format(total, epoch))
        for key, loss_i in total_dict.items():
            print("loss_{}: {:.3f} at epoch: {}".format(key, loss_i, epoch))

        if epoch % args.eval_interval == 0:
            print(f"----- evaluating at epoch {epoch} -----")
            model.eval()
            params = defaultdict(lambda: torch.tensor([]))

            with torch.no_grad():
                for i, (x, y) in tqdm(enumerate(test_loader)):
                    x = x.to(device)
                    y_pi, w_pi = model.clustering(x)
                    y_pred = torch.argmax(y_pi, -1)
                    w_pred = torch.argmax(w_pi, -1)

                    params["y"] = torch.cat([params["y"], y])
                    params["y_pred"] = torch.cat([params["y_pred"], y_pred.cpu()])
                    params["w_pred"] = torch.cat([params["w_pred"], w_pred.cpu()])

                y = params["y"].numpy().astype(int)
                y_pred = params["y_pred"].numpy().astype(int)
                w_pred = params["w_pred"].numpy().astype(int)

                plt.figure(figsize=(12, 8))
                cm = confusion_matrix(y, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
                plt.title(f"confusion matrix y / y' at epoch {epoch}")
                plt.savefig(f"cm_y_{epoch}.png")
                plt.close()

                plt.figure(figsize=(12, 8))
                cm = confusion_matrix(y, w_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
                plt.title(f"confusion matrix y / w' at epoch {epoch}")
                plt.savefig(f"cm_w_{epoch}.png")
                plt.close()


if __name__ == "__main__":
    main()
