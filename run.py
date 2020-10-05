import wandb
import hydra
import random
import torch
from tqdm import tqdm
import torchvision.transforms as tf

from src import get_net, get_model, get_optim, get_dataset, get_sampler, get_loader
from src import utils


def wandb_init(args):
    wandb.init(project=args.project,
               tags=args.tags,
               group=args.group,
               settings=wandb.Settings(symlink=False))
    wandb.run.name = args.name + '_' + wandb.run.id

def train(model, device, trainer, optim, epoch, use_amp=False):
    model.train()
    loss, num_samples = 0, 0
    with tqdm(total=len(trainer)) as pbar:
        for step, (x, target) in enumerate(trainer):
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            _, loss_step = model(x, target)
            optim.zero_grad()
            if use_amp:
                with amp.scale_loss(loss_step, optim) as loss_scaled:
                    loss_scaled.backward()
            else:
                loss_step.backward()
            optim.step()
            loss += loss_step.item()
            num_samples += x.shape[0]
            pbar.update(1)
    loss /= num_samples
    wandb.log({"epoch": epoch, "loss_train": loss})

def run(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    net = get_net(args.net.name, **args.net.params)

    model = get_model(args.model.name, net=net, **args.model.params).to(device)
    optim = get_optim(model.parameters(), args.optim.name, **args.optim.params)
    if args.use_amp:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Please install apex using...')
        model, optim = amp.initialize(model, optim, opt_level=args.opt_level)

    alt = -1 if args.use_other else None
    target_transform = utils.transforms.ToIndex(args.targets, alt)

    dataset = get_dataset(args.dataset.name, **args.dataset.params,
                          target_transform=target_transform)

    transform = tf.Compose([
        utils.transforms.SelectIndices(args.use_channels, 0),
        utils.transforms.CenterMaximizedResizeCrop(224),
    ])
    augment = tf.Compose([
        utils.transforms.SelectIndices(args.use_channels, 0),
        utils.transforms.RandomMaximizedResizeCrop(224),
    ])

    train_set, test_set = dataset.split(
        args.train_size, stratify=dataset.targets)
    train_set.transform = augment
    test_set.transform = transform

    sampler = get_sampler(args.sampler.name, dataset=train_set, **args.sampler.params)

    trainer = get_loader(train_set, sampler=sampler,
                         drop_last=True, **args.loader.params)
    tester = get_loader(test_set, **args.loader.params)

    for epoch in range(args.num_epochs):
        print(f"----- epoch: {epoch} -----")
        train(model, device, trainer, optim, epoch, use_amp=args.use_amp)


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    wandb_init(cfg.wandb)
    run(cfg.run)


if __name__ == "__main__":
    main()
