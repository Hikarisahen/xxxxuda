import os
import glob
from typing import Optional, Tuple, Callable

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class SatellitePatchDataset(Dataset):
    """Dataset for satellite image patches."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        image_size: int = 512,
        transform: Optional[Callable] = None,
        mask_suffix: str = "_mask",
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform
        self.mask_suffix = mask_suffix

        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif"))) + \
                          sorted(glob.glob(os.path.join(image_dir, "*.png"))) + \
                          sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = None

        if self.mask_dir and os.path.exists(self.mask_dir):
            mask_name = os.path.basename(image_path)
            mask_name = mask_name.replace(".tif", f"{self.mask_suffix}.tif")
            mask_name = mask_name.replace(".png", f"{self.mask_suffix}.png")
            mask_name = mask_name.replace(".jpg", f"{self.mask_suffix}.jpg")
            mask_path = os.path.join(self.mask_dir, mask_name)

            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L"))
                mask = (mask > 127).astype(np.float32)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        if mask is None:
            mask = torch.zeros(1, self.image_size, self.image_size)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def get_train_transforms(image_size: int = 512):
    return A.Compose([
        A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 512):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])