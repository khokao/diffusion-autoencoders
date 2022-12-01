import torchvision


def get_dataset(name, split, transform=None):
    """Get torchvision dataset.

    Args:
        name (str): Name of one of the following datasets,
            celeba: torchvision.datasets.CelebA
            lsun_{category}: torchvision.datasets.LSUN
        split (str): One of [`train`, `val`, `test`].
        transform (callable): A transform function that takes in an PIL image and returns a transformed version.

    Returns:
        dataset: A dataset class.
    """
    if name == 'celeba':
        dataset = torchvision.datasets.CelebA(
            root='./datasets/',
            split=split,
            target_type='attr',
            transform=transform,
            download=True,
        )
    else:
        raise NotImplementedError(f'Dataset name {name} if not supported.')

    return dataset
