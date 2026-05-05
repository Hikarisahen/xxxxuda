"""Paired (image, target) transforms for torchvision Mask R-CNN.

torchvision's high-level transforms.v2 supports this natively, but its API
shifts between torch versions. To stay version-agnostic we use the same
pattern as torchvision's reference detection scripts: each transform takes
and returns (image, target).
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def _img_to_tensor(img) -> torch.Tensor:
    if isinstance(img, torch.Tensor):
        return img
    arr = np.asarray(img)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = _img_to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image.flip(-1)
            if target.get("masks") is not None and target["masks"].numel() > 0:
                target["masks"] = target["masks"].flip(-1)
            if target.get("boxes") is not None and target["boxes"].numel() > 0:
                W = image.shape[-1]
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image.flip(-2)
            if target.get("masks") is not None and target["masks"].numel() > 0:
                target["masks"] = target["masks"].flip(-2)
            if target.get("boxes") is not None and target["boxes"].numel() > 0:
                H = image.shape[-2]
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = H - boxes[:, [3, 1]]
                target["boxes"] = boxes
        return image, target


class RandomColorJitter:
    """Image-only colour jitter. Boxes/masks are spatially invariant under it,
    so we don't need to touch the target."""
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1,
                 saturation: float = 0.1, prob: float = 0.5):
        self.b = brightness; self.c = contrast; self.s = saturation
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob and isinstance(image, torch.Tensor):
            # Brightness
            if self.b > 0:
                f = 1.0 + random.uniform(-self.b, self.b)
                image = (image * f).clamp_(0, 1)
            # Contrast: x' = (x - mean) * f + mean
            if self.c > 0:
                f = 1.0 + random.uniform(-self.c, self.c)
                mean = image.mean(dim=(-1, -2), keepdim=True)
                image = ((image - mean) * f + mean).clamp_(0, 1)
            # Saturation: blend toward greyscale
            if self.s > 0:
                f = 1.0 + random.uniform(-self.s, self.s)
                grey = image.mean(dim=-3, keepdim=True)
                image = (grey + (image - grey) * f).clamp_(0, 1)
        return image, target


def get_train_transforms() -> Compose:
    """Default training augmentation. Symmetric flips are safe for overhead
    imagery (rotation invariance is approximately a property of nadir views).
    """
    return Compose([
        ToTensor(),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        RandomColorJitter(0.1, 0.1, 0.1, prob=0.5),
    ])


def get_val_transforms() -> Compose:
    return Compose([ToTensor()])
