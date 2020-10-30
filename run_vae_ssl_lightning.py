import hydra
import pytorch_lightning as pl

from lightning import archs
from lightning import data
from lightning.utils import transforms
from lightning.data import samplers


@hydra.main(config_path="config", config_name="test")
def main(args):

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

    sampler_callback = lambda x: samplers.Balancer(x, args.sample_expansion)

    ds = data.Dataset(
        **args.dataset,
        transform=transform,
        augment_transform=augment_transform,
        target_transform=target_transform,
        sampler_callback=sampler_callback,
    )
    train_loader = ds.get_loader(**args.dataloader)
    test_loader = ds.get_loader(train=False, **args.dataloader)

    model = archs.M2()
    trainer = pl.Trainer(max_epochs=1000, gpus=-1)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
