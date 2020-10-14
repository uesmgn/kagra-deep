import wandb
import torch
import math
import matplotlib.pyplot as plt
import torchvision.transforms.functional as ttf
import torchvision as tv
import numpy as np
from sklearn.manifold import TSNE


class Plotter(object):
    def __init__(self, target):
        assert torch.is_tensor(target)
        if target.is_cuda:
            target = target.cpu()
        self.target = target.numpy()

    def confusion_matrix(self, y):
        assert torch.is_tensor(y)
        xlabels = list(np.unique(self.target))
        if y.is_cuda:
            y = y.cpu()
        if y.ndim == 1:
            assert isinstance(y, torch.LongTensor)
            y = y.numpy()
            ylabels = list(np.unique(y))
        elif y.ndim == 2:
            ylabels = list(range(y.shape[-1]))
            y = torch.argmax(y, -1)
            y = y.numpy()
        elif y.ndim == 3:
            print(y.shape)
            y = y[..., 0]
            print("select top head.")
            ylabels = list(range(y.shape[-1]))
            y = torch.argmax(y, -1)
            y = y.numpy()
        else:
            raise ValueError("Invalid input.")
        assert len(self.target) == len(y)

        cm = np.zeros((len(xlabels), len(ylabels)), dtype=np.int)
        for i, j in zip(self.target, y):
            cm[xlabels.index(i), ylabels.index(j)] += 1
        return wandb.plots.HeatMap(xlabels, ylabels, cm, show_text=True)


def plotter(target):
    return Plotter(target)


#
# def wandb_log(data, targets, name, epoch, type=None):
#
#     if type == "grid_image" and data.ndim == 4:
#         labels = np.unique(targets)
#         indices = {t: np.where(targets == t)[0] for t in labels}
#         idx = [v[0] for v in indices.values()]
#         if torch.max(data) > 1 or torch.min(data) < 0:
#             data = torch.sigmoid(data)
#         nrow = math.ceil(np.sqrt(len(idx))) + 1
#         data = tv.utils.make_grid(data[idx, ...], nrow=nrow)
#         pil = ttf.to_pil_image(data)
#         wandb.log({"epoch": epoch, name: [wandb.Image(pil, caption=name)]})
#
#     elif type == "tsne" and data.ndim == 2:
#         labels = np.unique(targets)
#         indices = {t: np.where(targets == t)[0] for t in labels}
#         z = TSNE(n_components=2).fit_transform(data)
#         xx, yy = z.T
#         for i, label in enumerate(labels):
#             idx = np.where(targets == label)
#             x = xx[idx]
#             y = yy[idx]
#             plt.scatter(x, y, label=label)
#         wandb.log({"epoch": epoch, name: plt})
#
#     elif type == "confusion_matrix" and data.ndim == 1:
#         xlabels = list(np.unique(targets))
#         ylabels = list(np.unique(data))
#         cm = np.zeros((len(xlabels), len(ylabels)), dtype=np.int)
#         for i, j in zip(data, targets):
#             cm[xlabels.index(i), ylabels.index(j)] += 1
#         wandb.log({"epoch": epoch, name: wandb.plots.HeatMap(xlabels, ylabels, cm, show_text=True)})
#
#
# #
# #
# # class EpochLogger:
# #     def __init__(self):
# #         self._log = defaultdict(lambda: [])
# #
# #     def update(self, name, value, epoch=None, verbose=False):
# #         if epoch is None:
# #             epoch = len(self._log[name])+1
# #         self._log[name].append((epoch, value))
# #         if verbose:
# #             if isinstance(value, float):
# #                 print(f'{name} at epoch {epoch} = {value:.3f}')
# #             else:
# #                 print(f'{name} at epoch {epoch} = {value}')
# #
# #     def __getitem__(self, name):
# #         log = self._log[name]
# #         return np.array(log).T
# #
# #     def get_plot(self, name):
# #         fig, ax = plt.subplots(figsize=(8,6))
# #         xx, yy = self[name]
# #         ax.set_xlim([min(xx), max(xx)])
# #         ax.set_xlabel('epoch')
# #         ax.set_ylabel(name)
# #         ax.xaxis.set_major_locator(ticker.MaxNLocator(10, integer=True))
# #         ax.plot(xx, yy)
# #         return fig, ax
# #
# #     def get_plots(self):
# #         for k, _ in self._log.items():
# #             fig, ax = self.get_plot(k)
# #             yield k, fig, ax
# #
# # class Evaluator:
# #     def __init__(self):
# #         self._log = defaultdict(lambda: [])
# #
# #     def update(self, name, tensor):
# #         self._log[name].append(tensor.cpu())
# #
# #     def __getitem__(self, name):
# #         log = self._log[name]
# #         tensor = torch.cat(log)
# #         return tensor.numpy().squeeze()
# #
# #     def confusion_matrix(self, xlabel, ylabel, xaxis=None, yaxis=None, idx=None):
# #         xx, yy = self[xlabel], self[ylabel]
# #         assert xx.ndim < 3 and yy.ndim < 3
# #         xaxis = xaxis or sorted(np.unique(xx))
# #         yaxis = yaxis or sorted(np.unique(yy))
# #         if xx.ndim == 2:
# #             xx = stats.mode(xx, axis=0).mode.squeeze() if idx is None else xx[idx]
# #         if yy.ndim == 2:
# #             yy = stats.mode(yy, axis=0).mode.squeeze() if idx is None else yy[idx]
# #         cm = np.zeros((len(xaxis), len(yaxis)), dtype=np.int)
# #         for i, j in zip(xx, yy):
# #             cm[xaxis.index(i), yaxis.index(j)] += 1
# #         return cm.T, xaxis, yaxis
# #
# #     def get_confusion_matrix(self, xlabel, ylabel, xaxis=None, yaxis=None, idx=None):
# #         if isinstance(xaxis, list):
# #             xlabels = xaxis
# #             xaxis = list(range(len(xaxis)))
# #         elif isinstance(xaxis, int):
# #             xlabels = list(range(xaxis))
# #             xaxis = xlabels
# #         if isinstance(yaxis, list):
# #             ylabels = yaxis
# #             yaxis = list(range(len(yaxis)))
# #         elif isinstance(yaxis, int):
# #             ylabels = list(range(yaxis))
# #             yaxis = ylabels
# #         cm, xaxis, yaxis = self.confusion_matrix(xlabel, ylabel, xaxis, yaxis, idx)
# #         figsize=(len(xaxis) // 2.5, len(yaxis) // 2.5)
# #         fig, ax = plt.subplots(figsize=figsize)
# #         ax.set_xlabel(xlabel)
# #         ax.set_ylabel(ylabel)
# #         cmap = plt.get_cmap('Purples')
# #         cm_norm = sklearn.preprocessing.normalize(cm, axis=1, norm='l1')
# #         ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, origin='lower')
# #         ax.set_xticks(np.arange(len(xlabels)))
# #         ax.set_yticks(np.arange(len(ylabels)))
# #         ax.set_xticklabels(xlabels)
# #         ax.set_yticklabels(ylabels)
# #         plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# #         for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#             num = "{}".format(cm[i, j])
#             color = "white" if cm_norm[i, j] > 0.5 else "black"
#             ax.text(j, i, num, fontsize=10, color=color, ha='center', va='center')
#         fig.tight_layout()
#         return fig, ax
#
#     def get_latent_features(self, z_key: str, label_key: str, targets: list):
#         z = self[z_key]
#         labels = self[label_key]
#         assert len(labels) == len(z)
#         assert z.ndim == 2
#         if z.shape[-1] > 2:
#             z = TSNE(n_components=2).fit_transform(z)
#         xx, yy = z.T
#         fig, ax = plt.subplots(figsize=(8, 8))
#         for i, target in enumerate(targets):
#             if i in labels:
#                 idx = np.where(labels==i)
#             elif target in labels:
#                 idx = np.where(labels==target)
#             else:
#                 continue
#             x = xx[idx]
#             y = yy[idx]
#             ax.scatter(x, y, s=8.0, label=target)
#         ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
#                   borderaxespad=0, fontsize=8)
#         fig.tight_layout()
#         return fig, ax
