import wandb
import hydra
import logging
import torch
from collections import abc

try:
    import apex

    use_apex = True
except ImportError:
    use_apex = False

logger = logging.getLogger(__name__)


def init_wandb(args):
    wandb.init(project=args.project, tags=args.tags, group=args.group)
    wandb.run.name = args.name + "_" + wandb.run.id


def init_config(args):
    from src import config
    from src.utils import transforms as tf
    from src.data import samplers

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


def to_device(*args, device="cpu"):
    ret = []
    for arg in args:
        if torch.is_tensor(arg):
            ret.append(arg.to(device, non_blocking=True))
        elif isinstance(arg, abc.Sequence):
            for x in arg:
                ret += to_device(x)
        else:
            raise ValueError(f"Input is invalid argument type: {type(arg)}.")
    return tuple(ret)


def train(model, optim, loader, device="cpu", weights=None, use_apex=False):
    model.train()
    weights = torch.tensor(weights).to(device)
    loss = 0
    num_samples = 0
    for data in loader:
        data = to_device(*data, device=device)
        l = model(*data)
        loss_step = (l * weights).sum()
        optim.zero_grad()
        if use_apex:
            with apex.amp.scale_loss(loss_step, optim) as loss_scaled:
                loss_scaled.backward()
        else:
            loss_step.backward()
        optim.step()
        loss += l.detach().cpu()
        num_samples += data[0].shape[0]
    loss /= num_samples
    return loss


@hydra.main(config_path="config", config_name="test")
def main(args):
    global use_apex
    use_apex = args.use_apex and use_apex

    init_wandb(args.wandb)

    cfg = init_config(args)

    device = args.device
    num_epochs = args.num_epochs
    weights = args.weights
    net = cfg.get_net(args.net)
    model = cfg.get_model(args.model, net=net).to(device)
    optim = cfg.get_optim(args.optim, params=model.parameters())
    train_set, test_set = cfg.get_datasets(args.dataset)
    train_loader = cfg.get_loader(args.train, train_set)
    test_loader = cfg.get_loader(args.test, test_set)

    for epoch in range(num_epochs):
        print(f"--- training at epoch {epoch} ---")
        loss_train = train(
            model, optim, train_loader, device=device, weights=weights, use_apex=use_apex
        )
        print(loss)
        wandb.log({"epoch": epoch, "loss_train": loss_train})


if __name__ == "__main__":
    main()
