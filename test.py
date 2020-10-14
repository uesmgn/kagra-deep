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
from src.utils import transforms as tf
from src.utils import stats
from src.utils.functional import to_device, tensordict
from src.data import samplers


logger = logging.getLogger(__name__)


def wandb_init(args):
    wandb.init(project=args.project, tags=args.tags, group=args.group)
    wandb.run.name = args.name + "_" + wandb.run.id


def config_init(args):
    transform = tf.Compose(
        [
            tf.SelectIndices(args.use_channels, 0),
            tf.CenterMaximizedResizeCrop(224),
        ]
    )
    augment_transform = tf.Compose(
        [
            tf.SelectIndices(args.use_channels, 0),
            tf.RandomMaximizedResizeCrop(224),
        ]
    )
    alt = -1 if args.use_other else None
    target_transform = tf.ToIndex(args.targets, alt=alt)

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


def train(model, optim, loader, device, weights=None, use_apex=False):
    model.train()
    loss = 0
    num_samples = 0
    with tqdm(total=len(loader)) as pbar:
        for data in loader:
            data = to_device(device, *data)
            l = model(*data)
            loss_step = (l * weights).sum()
            optim.zero_grad()
            if use_apex:
                with amp.scale_loss(loss_step, optim) as loss_scaled:
                    loss_scaled.backward()
            else:
                loss_step.backward()
            optim.step()
            loss += l.detach()
            num_samples += data[0].shape[0]
            pbar.update(1)
    loss /= num_samples
    return loss


def eval(model, loader, device):
    model.eval()
    params = tensordict()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for data in loader:
                data = to_device(device, *data)
                l, p = model(*data)
                params.stack(p)
                loss += l
                num_samples += data[0].shape[0]
                pbar.update(1)
    loss /= num_samples
    return loss, params


def log_loss(loss, epoch, prefix="loss"):
    d = {}
    if torch.is_tensor(loss):
        loss = loss.squeeze()
        if loss.is_cuda:
            loss = loss.cpu()
        if loss.ndim == 0:
            loss = loss.item()
            d = {prefix: loss}
        elif loss.ndim == 1:
            loss = loss.numpy()
            total = loss.sum()
            d = {f"{prefix}_{i}": l for i, l in enumerate(loss)}
            d[f"{prefix}_total"] = total
        else:
            raise ValueError("Invalid arguments.")
    elif isinstance(loss, numbers.Number):
        d = {prefix: loss}
    else:
        raise ValueError("Invalid arguments.")
    wandb.log(d, step=epoch)


def log_params(params, epoch, cfg):
    target = params.pop("target")
    plt = stats.plotter(targets)
    for k, v in params.items():
        if k in cfg:
            func = cfg[k]
            if isinstance(func, abc.Sequence):
                for f in func:
                    obj = getattr(plt, f)(v)
                    wandb.log({f"{k}_{f}": obj}, step=epoch)
            else:
                obj = getattr(plt, func)(v)
                wandb.log({f"{k}_{func}": obj}, step=epoch)


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
    train_set, test_set = cfg.get_datasets(args.dataset)
    train_loader = cfg.get_loader(args.train, train_set)
    test_loader = cfg.get_loader(args.test, test_set)

    for epoch in range(num_epochs):
        logger.info(f"--- training at epoch {epoch} ---")
        train_loss = train(model, optim, train_loader, device, weights=weights, use_apex=use_apex)
        log_loss(train_loss, epoch, prefix="train_loss")
        if epoch % args.eval_step == 0:
            logger.info(f"--- evaluating at epoch {epoch} ---")
            test_loss, params = eval(model, test_loader, device)
            log_loss(test_loss, epoch, prefix="test_loss")
            log_params(params, epoch, args.log)


if __name__ == "__main__":
    main()
