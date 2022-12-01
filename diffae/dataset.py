import torchvision
from PIL import Image


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


def get_torchvision_transforms(cfg, mode):
    assert mode in {'train', 'test'}
    if mode == 'train':
        transforms_cfg = cfg['train']['dataset']
    else:
        transforms_cfg = cfg['test']['dataset']

    transforms = []
    for t in transforms_cfg:
        if hasattr(torchvision.transforms, t['name']):
            transform_cls = getattr(torchvision.transforms, t['name'])(**t['params'])
        else:
            raise ValueError(f'Tranform {t["name"]} is not defined')
        transforms.append(transform_cls)
    transforms = torchvision.transforms.Compose(transforms)

    return transforms


def load_image_pillow(image_path):
    with Image.open(image_path) as img:
        image = img.convert('RGB')
    return image
