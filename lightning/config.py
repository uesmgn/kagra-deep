from .data import datasets


def get_dataset(
    root,
    type="basic",
    transform=None,
    target_transform=None,
    augment_transform=None,
    sampler_callback=None,
    train_size=0.8,
    labeled_size=0.8,
):

    dataset = datasets.HDF5(root=root, transform=transform, target_transform=target_transform)
    train_set, test_set = dataset.split(train_size, stratify=dataset.targets)

    if type == "basic":
        if callable(sampler_callback):
            train_set.transform = augment_transform
    elif type == "ss":
        if callable(sampler_callback):
            train_set.transform = augment_transform
        l, u = train_set.split(labeled_size, stratify=train_set.targets)
        train_set = (l, u)
    elif type == "co":
        train_set = datasets.Co(train_set, augment_transform)
    elif type == "co+ss":
        l, u = train_set.split(labeled_size, stratify=train_set.targets)
        l, u = map(lambda x: datasets.Co(x, augment_transform), (l, u))
        train_set = (l, u)
    return train_set, test_set
