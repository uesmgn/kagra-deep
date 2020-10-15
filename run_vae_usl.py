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
from src.utils import metrics
from src.utils.functional import to_device, flatten, tensordict
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


def train(model, optim, loader, device, epoch, weights=1.0, use_apex=False):
    # training loss for each items e.g. BCE, KL, CE
    res = tensordict({"epoch": epoch})

    model.train()
    for data in tqdm(loader):
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

        res.stack({"train_loss": loss})

    res.mean("train_loss", keep_dim=-1).flatten("train_loss")
    return res


def eval(model, loader, device, epoch, metrics_callback=None):
    # loss for each element
    res = tensordict({"epoch": epoch})
    params = tensordict()

    model.eval()
    with torch.no_grad():
        for data in tqdm(loader):
            data = to_device(device, *data)
            loss, param = model(*data)

            res.stack({"eval_loss": loss})
            params.cat(param)

    res.mean("eval_loss", keep_dim=-1).flatten("eval_loss")

    if callable(metrics_callback):
        res.update(metrics_callback(**params))
    elif isinstance(metrics_callback, abc.Sequence):
        for fn in metrics_callback:
            if callable(fn):
                res.update(fn(**params))

    return res


@hydra.main(config_path="config", config_name="vae_usl")
def main(args):
    wandb_init(args.wandb)
    wandb.config.update(flatten(args))

    global use_apex
    use_apex = args.use_apex and use_apex

    cfg = config_init(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    num_epochs = args.num_epochs

    metrics_fn = None
    if isinstance(args.metrics_fn, abc.Sequence):
        metrics_fn = [getattr(metrics, fn) for fn in args.metrics_fn if hasattr(metrics, fn)]
    elif hasattr(metrics, args.metrics_fn):
        metrics_fn = getattr(metrics, args.metrics_fn)

    if isinstance(args.weights, abc.Sequence):
        weights = torch.tensor(args.weights).to(device)
    elif isinstance(args.weights, numbers.Number):
        weights = float(args.weights)
    else:
        weights = 1.0

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
        train_res = train(
            model, optim, train_loader, device, epoch, weights=weights, use_apex=use_apex
        )
        wandb.log(train_res)
        if epoch % args.eval_step == 0:
            logger.info(f"--- evaluating at epoch {epoch} ---")
            eval_res = eval(model, eval_loader, device, epoch, metrics_fn)
            wandb.log(eval_res)


if __name__ == "__main__":
    main()
