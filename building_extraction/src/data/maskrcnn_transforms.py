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


class RandomRotation90:
    """Rotate image + masks by k*90 degrees, k uniform in {0,1,2,3}.

    Safe for nadir overhead imagery — rotation invariance approximately holds.
    Combined with H/V flips this gives the full 8-element dihedral group, which
    is the right symmetry for satellite tiles. Boxes are recomputed from the
    rotated masks via torchvision.ops.masks_to_boxes (more robust than rotating
    boxes directly, especially when k=1 or 3 swaps H<->W).
    """
    def __call__(self, image, target):
        k = random.randint(0, 3)
        if k == 0:
            return image, target
        image = torch.rot90(image, k=k, dims=(-2, -1))
        masks = target.get("masks")
        if masks is not None and masks.shape[0] > 0:
            masks = torch.rot90(masks, k=k, dims=(-2, -1)).contiguous()
            target["masks"] = masks
            from torchvision.ops import masks_to_boxes
            target["boxes"] = masks_to_boxes(masks)
        return image, target


class RandomGamma:
    """Image-only gamma correction. Simulates exposure differences between
    sensors / acquisition times — a meaningful axis of Vaihingen↔Potsdam
    domain gap that plain brightness scaling doesn't cover."""
    def __init__(self, gamma_range: tuple = (0.7, 1.4), prob: float = 0.5):
        self.lo, self.hi = gamma_range
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob and isinstance(image, torch.Tensor):
            g = random.uniform(self.lo, self.hi)
            image = image.clamp(0, 1).pow(g)
        return image, target


class RandomGaussianBlur:
    """Image-only Gaussian blur with random sigma. Helps the model tolerate
    GSD / focus / atmospheric differences between domains."""
    def __init__(self, sigma_range: tuple = (0.1, 1.5), kernel_size: int = 5,
                 prob: float = 0.3):
        self.lo, self.hi = sigma_range
        self.k = kernel_size  # must be odd
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob and isinstance(image, torch.Tensor):
            sigma = random.uniform(self.lo, self.hi)
            from torchvision.transforms.functional import gaussian_blur
            image = gaussian_blur(image, kernel_size=[self.k, self.k],
                                  sigma=[sigma, sigma])
        return image, target


def get_train_transforms() -> Compose:
    """Stronger augmentation for cross-domain (Vaihingen → Potsdam) training.

    Why this stack:
      * H/V flip + 90° rotation: full dihedral group, the natural symmetry of
        nadir overhead imagery — multiplies effective spatial diversity 8x.
      * ColorJitter at 0.4 (was 0.1, basically a no-op): bridges the photometric
        gap between the two sensors / acquisition conditions.
      * RandomGamma: exposure shift axis ColorJitter doesn't cover.
      * GaussianBlur (low prob): tolerance to GSD / focus differences.

    The previous defaults were so weak they barely contributed; the source
    model was effectively training without augmentation — a major reason
    cross-domain generalisation was poor.
    """
    return Compose([
        ToTensor(),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        RandomRotation90(),
        RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, prob=0.8),
        RandomGamma(gamma_range=(0.7, 1.4), prob=0.5),
        RandomGaussianBlur(sigma_range=(0.1, 1.5), kernel_size=5, prob=0.3),
    ])


def get_val_transforms() -> Compose:
    return Compose([ToTensor()])
