"""COCO-format building instance dataset for torchvision Mask R-CNN.

Each __getitem__ returns (image, target) where:
    image  : torch.FloatTensor [3, H, W] in [0, 1]
    target : dict with keys
        boxes    : [N, 4] float32  (x1, y1, x2, y2)
        labels   : [N]    int64    (1 for building; 0 reserved for background)
        masks    : [N, H, W] uint8 (binary)
        image_id : scalar int64
        area     : [N]    float32
        iscrowd  : [N]    int64

Why this lives separate from `SatellitePatchDataset`:
    The SemSeg dataset returns a single union mask per image. Instance models
    need per-instance masks + boxes + labels. These two formats can't share an
    implementation cleanly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from pycocotools.coco import COCO
except ImportError as e:
    raise ImportError(
        "pycocotools is required. Install with `pip install pycocotools`."
    ) from e


class CocoBuildingDataset(Dataset):
    """Wraps a COCO JSON + image directory for instance-segmentation training.

    Args:
        ann_file: path to COCO-format JSON (e.g. annotations/vaihingen_train_coco.json).
        image_dir: directory containing the images referenced by `file_name`.
        transforms: optional callable taking (image, target) and returning the
            transformed pair. Use scripts/maskrcnn_transforms.py for paired
            box/mask-aware augmentation.
        require_anns: if True (default), images with zero non-crowd annotations
            are filtered out — the model has nothing to learn from them and they
            sometimes trip torchvision's empty-target handling.
        category_id: which COCO category id corresponds to "building". Our
            converter writes category_id=0; we map that to label 1 (since 0 is
            reserved for background in torchvision detection models).
    """

    def __init__(
        self,
        ann_file: str,
        image_dir: str,
        transforms: Optional[Callable] = None,
        require_anns: bool = True,
        category_id: int = 0,
    ) -> None:
        super().__init__()
        self.coco = COCO(ann_file)
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.category_id = category_id

        all_ids = sorted(self.coco.imgs.keys())
        if require_anns:
            self.image_ids: List[int] = [
                i for i in all_ids
                if len(self.coco.getAnnIds(imgIds=i, iscrowd=False)) > 0
            ]
        else:
            self.image_ids = all_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, img_info: dict) -> Image.Image:
        path = self.image_dir / img_info["file_name"]
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img = self._load_image(img_info)
        H, W = img.height, img.width

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []
        areas: List[float] = []

        for ann in anns:
            if ann.get("category_id", self.category_id) != self.category_id:
                continue
            mask = self.coco.annToMask(ann)
            if mask.sum() < 4:
                continue
            ys, xs = np.where(mask > 0)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(1)  # 1 = building (0 = background, reserved)
            masks.append(mask.astype(np.uint8))
            areas.append(float(ann.get("area", mask.sum())))

        if boxes:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.from_numpy(np.stack(masks)).to(torch.uint8),
                "image_id": torch.tensor(img_id, dtype=torch.int64),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            }
        else:
            # Defensive empty target — should be rare since require_anns=True
            # filters most cases, but a polygon-degenerate ann can still slip in.
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
                "image_id": torch.tensor(img_id, dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            # Plain ToTensor when no transforms supplied.
            img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0

        return img, target


def collate_fn(batch):
    """torchvision detection models want lists, not stacked tensors,
    because images can have different sizes."""
    return tuple(zip(*batch))
