"""
The codes are modified.

Link:
    - [Meter] https://github.com/Megvii-BaseDetection/YOLOX/
      blob/a5bb5ab12a61b8a25a5c3c11ae6f06397eb9b296/yolox/utils/metric.py#L54-L98
"""
import os
import random
from collections import deque

import numpy as np
import torch
import torchvision


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_reproducibility_cudnn():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Meter:
    def __init__(self):
        self._deque = deque()
        self._count = 0
        self._total = 0.0

    def update(self, value):
        self._deque.append(value)
        self._count += 1
        self._total += value

    def reset(self):
        self._deque.clear()
        self._count = 0
        self._total = 0.0

    @property
    def avg(self):
        d = np.array(list(self._deque))
        return d.mean()

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None


def get_torchvision_unnormalize(mean, std):
    """
    Get unnormalize function. reference: https://github.com/pytorch/vision/issues/528

    Args:
        mean, std (list): Normalization parameters (RGB)

    Returns:
        unnormalize (torchvision.transforms.Normalize): Unnormalize function.
    """
    assert len(mean) == 3
    assert len(std) == 3
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)

    unnormalize = torchvision.transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return unnormalize
