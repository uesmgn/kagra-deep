import wandb
import hydra
import logging
import torch
from collections import abc
import numbers
from tqdm import tqdm

try:
    from apex import amp

    use_apex = True
except ImportError:
    use_apex = False

from src import config
from src.utils import transforms
from src.utils import stats
from src.utils.functional import to_device, tensordict, multi_class_metrics
from src.data import samplers


logger = logging.getLogger(__name__)


def wandb_init(args):
    wandb.init(project=args.project, tags=args.tags, group=args.group)
    wandb.run.name = args.name + "_" + wandb.run.id


def config_init(args):
    transform = transforms.Compose(
        [
            transforms.SelectIndices(args.use_channels, 0),
            transforms.CenterMaximizedResizeCrop(224),
        ]
    )
    augment_transform = transforms.Compose(
        [
            transforms.SelectIndices(args.use_channels, 0),
            transforms.RandomMaximizedResizeCrop(224),
        ]
    )
    alt = -1 if args.use_other else None
    target_transform = transforms.ToIndex(args.targets, alt=alt)

    def sampler_callback(ds):
        return samplers.Balancer(ds, args.sample_expansion)

    params = {
        "type": args.type,
        "transform": transform,
        "augment_transform": augment_transform,
        "target_transform": target_transform,
        "sampler_callback": sampler_callback,
    }

    return config.Config(**params)


def train(model, loader, device, optim, weights=1.0, use_apex=False):

    result = tensordict()

    model.train()

    with tqdm(total=len(loader)) as pbar:
        for data in loader:
            data = to_device(device, *data)
            loss = model(*data)
            loss_step = (loss * weights).sum()

            optim.zero_grad()
            if use_apex:
                with amp.scale_loss(loss_step, optim) as loss_scaled:
                    loss_scaled.backward()
            else:
                loss_step.backward()
            optim.step()

            result.cat({"train_loss": loss})
            pbar.update(1)
    result.reduction("mean", keep_dim=-1)
    result.flatten()
    return result


def eval(model, loader, device):
    result = tensordict()
    params = tensordict()

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for data in loader:
                data = to_device(device, *data)
                loss, t, p = model(*data)
                result.cat({"eval_loss": loss})
                params.cat({"target": t, "pred": p}, dim=0)
                pbar.update(1)

    metrics = multi_class_metrics(params["target"], params["pred"])
    result.reduction("mean", keep_dim=-1).flatten()
    return result, metrics


# def log_loss(epoch, params, prefix=""):
#     d = {}
#     if torch.is_tensor(loss):
#         loss = loss.squeeze()
#         if loss.is_cuda:
#             loss = loss.cpu()
#         if loss.ndim == 0:
#             total = loss.item()
#         elif loss.ndim == 1:
#             losses = loss.numpy()
#             total = losses.sum()
#             for i, l in enumerate(losses):
#                 key = "_".join(filter(bool, [prefix, str(i)]))
#                 d[key] = l
#         else:
#             raise ValueError("Invalid arguments.")
#     elif isinstance(loss, numbers.Number):
#         total = loss
#     else:
#         raise ValueError("Invalid arguments.")
#     key = "_".join(filter(bool, [prefix, "total"]))
#     d[key] = total
#     wandb.log(d, step=epoch)
#
#
# def log_params(epoch, params, funcs, prefix=""):
#     target = params.pop("target")
#     plt = stats.plotter(target)
#
#     for key, param in params.items():
#         f = None
#
#         if key in funcs:
#             f = funcs[key]
#
#         if isinstance(f, str):
#             name = "_".join(map(lambda x: str(x), filter(bool, [prefix, key, f])))
#             obj = None
#
#             if f == "confusion_matrix":
#                 xlabels, ylabels, matrix = plt.confusion_matrix(param)
#                 try:
#                     obj = wandb.plots.HeatMap(xlabels, ylabels, matrix, show_text=True)
#                 except:
#                     print(name, matrix)
#
#             if obj is not None:
#                 try:
#                     wandb.log({name: obj, "epoch": epoch})
#                 except:
#                     pass
#         else:
#             pass


@hydra.main(config_path="config", config_name="test")
def main(args):
    global use_apex
    use_apex = args.use_apex and use_apex

    wandb_init(args.wandb)

    cfg = config_init(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    num_epochs = args.num_epochs

    weights = args.weights
    if isinstance(weights, abc.Sequence):
        weights = torch.tensor(weights).to(device)
    elif isinstance(weights, numbers.Number):
        weights = float(weights)
    else:
        raise ValueError(f"Invalid weights: {weights}")

    net = cfg.get_net(args.net)
    model = cfg.get_model(args.model, net=net).to(device)
    optim = cfg.get_optim(args.optim, params=model.parameters())
    if use_apex:
        model, optim = amp.initialize(model, optim, opt_level=args.opt_level)
    train_set, eval_set = cfg.get_datasets(args.dataset)
    train_loader = cfg.get_loader(args.train, train_set)
    eval_loader = cfg.get_loader(args.eval, eval_set, train=False)

    for epoch in range(num_epochs):
        logger.info(f"--- training at epoch {epoch} ---")
        train_loss = train(model, train_loader, device, optim, weights=weights, use_apex=use_apex)
        wandb.log(train_loss, step=epoch)
        if epoch % args.eval_step == 0:
            logger.info(f"--- evaluating at epoch {epoch} ---")
            eval_loss, eval_metrics = eval(model, eval_loader, device)
            wandb.log(eval_loss, step=epoch)
            wandb.log(eval_metrics, step=epoch)


if __name__ == "__main__":
    main()
