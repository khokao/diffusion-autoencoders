
import csv
import os

import PIL
import torch
import torchvision
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg


class EmbeddingDataset(Dataset):
    def __init__(self, image_dataset, encoder, cfg):
        """
        Args:
            image_dataset ()
        """
        super().__init__()
        self.device = cfg['general']['device']
        self.style_embs, self.targets = self._extract_embedding(image_dataset, encoder)

    @torch.inference_mode()
    def _extract_embedding(self, image_dataset, encoder):
        encoder.to(self.device)
        encoder.eval()

        style_embs = []
        targets = []
        for batch in tqdm(image_dataset):
            image, target = batch
            image = image.to(self.device)
            style_emb = encoder(image)
            style_embs.append(style_emb.to('cpu'))
            targets.append(target)
        style_embs = torch.cat(style_embs)
        targets = torch.cat(targets)

        return style_embs, targets

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        style_emb = self.style_embs[index]
        target = self.targets[index]
        return style_emb, target.long()


class CelebAHQ(CelebA):
    base_folder = 'celebahq'
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ('1badu11NqxGf6qM3PTTooQDJvQbejgbTv', 'b08032b342a8e0cf84c273db2b52eef3', 'CelebAMask-HQ.zip'),
        ('0B7EVK8r0v71pY0NSMzRuSXJEVkk', 'd32c9cbf5e040fd4025c592c306e6668', 'list_eval_partition.txt'),
    ]

    def __init__(
        self,
        root,
        split='train',
        target_type='attr',
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(CelebA, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        split_map = {
            'train': 0,
            'valid': 1,
            'test': 2,
            'all': None,
        }
        split_ = split_map[verify_str_arg(split.lower(), 'split', ('train', 'valid', 'test', 'all'))]
        splits = self._load_csv('list_eval_partition.txt')
        id_map = self._load_id_map('CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt', header=0)
        attr = self._load_csv('CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt', header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
            self.filename = [id_map[f] for f in self.filename if f in id_map.keys()]

        if split_ is not None:
            mask = torch.tensor([False] * attr.data.shape[0])
            from tqdm import tqdm
            for i, (celeba_id, celebahq_id) in tqdm(enumerate(id_map.items())):
                assert f'{i}.jpg' == celebahq_id
                if splits.data[splits.index.index(celeba_id)] == split_:
                    mask[i] = True

        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor')
        self.attr_names = attr.header

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in ['.zip', '.7z'] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, 'CelebAMask-HQ/CelebA-HQ-img'))

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        extract_archive(os.path.join(self.root, self.base_folder, 'CelebAMask-HQ.zip'))

    def _load_id_map(self, filename, header=None):
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[2] for row in data]  # orig_file

        id_map = {}
        for idx, orig_file in zip(indices, data):
            assert isinstance(orig_file, str)
            id_map[orig_file] = f'{idx}.jpg'

        return id_map

    def __getitem__(self, index: int):
        x = PIL.Image.open(
            os.path.join(self.root, self.base_folder, 'CelebAMask-HQ/CelebA-HQ-img', self.filename[index])
        )

        target = []
        for t in self.target_type:
            if t == 'attr':
                target.append(self.attr[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            x = self.transform(x)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return x, target


def get_dataset(name, split, transform=None):
    """Get torchvision dataset.

    Args:
        name (str): Name of one of the following datasets,
            celeba: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
            celebahq: http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html
        split (str): One of [`train`, `val`, `test`].
        transform (callable): A transform function that takes in an PIL image and returns a transformed version.

    Returns:
        dataset: A dataset class.
    """
    if name == 'celeba':
        dataset = CelebA(
            root='./datasets/',
            split=split,
            target_type='attr',
            transform=transform,
            download=True,
        )
    elif name == 'celebahq':
        dataset = CelebAHQ(
            root='./datasets/',
            split=split,
            target_type='attr',
            transform=transform,
            download=True,
        )
    else:
        raise NotImplementedError(f'Dataset name `{name}` is not supported.')

    return dataset


class D2CCrop:
    def __init__(self):
        cx = 89
        cy = 121
        self.x1 = cy - 64
        self.x2 = cy + 64
        self.y1 = cx - 64
        self.y2 = cx + 64

    def __call__(self, img):
        img = torchvision.transforms.functional.crop(
            img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1,
        )
        return img

    def __repr__(self):
        return self.__class__.__name__ + f'(x1={self.x1}, x2={self.x2}, y1={self.y1}, y2={self.y2}'


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
        elif t['name'] == 'D2CCrop':
            # For CerebA (not CelabA-HQ) dataset, D2C Crop is applied first.
            transform_cls = D2CCrop()
        else:
            raise ValueError(f'Tranform {t["name"]} is not defined')
        transforms.append(transform_cls)
    transforms = torchvision.transforms.Compose(transforms)

    return transforms


def load_image_pillow(image_path):
    with Image.open(image_path) as img:
        image = img.convert('RGB')
    return image
