import wandb
import torch
import math
import matplotlib.pyplot as plt
import torchvision.transforms.functional as ttf
import torchvision as tv
import numpy as np
from sklearn.manifold import TSNE


def wandb_log(data, targets, name, epoch, type=None):

    if type == "grid_image" and data.ndim == 4:
        labels = np.unique(targets)
        indices = {t: np.where(targets==t)[0] for t in labels}
        idx = [v[0] for v in indices.values()]
        if torch.max(data) > 1 or torch.min(data) < 0:
            data = torch.sigmoid(data)
        nrow = math.ceil(np.sqrt(len(idx))) + 1
        data = tv.utils.make_grid(data[idx, ...], nrow=nrow)
        pil = ttf.to_pil_image(data)
        wandb.log({"epoch": epoch, name: [wandb.Image(pil, caption=name)]})

    elif type == "tsne" and data.ndim == 2:
        labels = np.unique(targets)
        indices = {t: np.where(targets==t)[0] for t in labels}
        z = TSNE(n_components=2).fit_transform(data)
        xx, yy = z.T
        for i, label in enumerate(labels):
            idx = np.where(targets==label)
            x = xx[idx]
            y = yy[idx]
            plt.scatter(x, y, label=label)
        wandb.log({"epoch": epoch, name: plt})

    elif type == "confusion_matrix" and data.ndim == 1:
        xlabels = list(np.unique(targets))
        ylabels = list(np.unique(data))
        cm = np.zeros((len(xlabels), len(ylabels)), dtype=np.int)
        for i, j in zip(data, targets):
            cm[xlabels.index(i), ylabels.index(j)] += 1
        wandb.log({
            "epoch": epoch,
            name: wandb.plots.HeatMap(xlabels, ylabels, cm, show_text=True)
        })
