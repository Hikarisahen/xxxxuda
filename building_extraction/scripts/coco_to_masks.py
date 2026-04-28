#!/usr/bin/env python3
"""Rasterise a COCO instance-segmentation JSON into per-image binary masks.

Lays the result out the way configs/unet_finetune.yaml expects:

    <out_root>/
      train_images/<file_name>.png      (symlinks/copies of the source images)
      train_masks/<stem>_mask.png       (uint8 0/255 binary mask)
      val_images/...
      val_masks/...

Usage:
    python scripts/coco_to_masks.py \
        --train-json annotations/vaihingen_train_coco.json \
        --val-json   annotations/vaihingen_val_coco.json \
        --image-root <DIR with the original Vaihingen tiles> \
        --out-root   data
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def rasterise_split(coco_path: Path, image_root: Path, out_images: Path,
                    out_masks: Path, copy_images: bool) -> None:
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    with open(coco_path, "r") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    skipped = 0
    for img_id, img_info in tqdm(images_by_id.items(), desc=coco_path.stem):
        file_name = img_info["file_name"]
        h, w = img_info["height"], img_info["width"]

        src = image_root / file_name
        if not src.exists():
            # try basename match in case file_name has subdirs
            src = image_root / Path(file_name).name
        if not src.exists():
            skipped += 1
            continue

        # Image: copy or symlink into the expected layout.
        dst_img = out_images / Path(file_name).name
        if not dst_img.exists():
            if copy_images:
                shutil.copy2(src, dst_img)
            else:
                try:
                    dst_img.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dst_img)  # Windows w/o symlink perms

        # Mask: rasterise all polygon annotations for this image.
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns_by_image.get(img_id, []):
            for poly in ann.get("segmentation", []):
                if len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 255)

        stem = Path(file_name).stem
        Image.fromarray(mask).save(out_masks / f"{stem}_mask.png")

    if skipped:
        print(f"[{coco_path.name}] WARNING: {skipped} images not found under {image_root}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-json", type=Path, required=True)
    p.add_argument("--val-json", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True,
                   help="Directory containing the original Vaihingen image tiles.")
    p.add_argument("--out-root", type=Path, default=Path("data"))
    p.add_argument("--copy", action="store_true",
                   help="Copy images instead of symlinking (use on Windows w/o admin).")
    args = p.parse_args()

    rasterise_split(args.train_json, args.image_root,
                    args.out_root / "train_images",
                    args.out_root / "pseudo_labels",   # finetune yaml's train_masks key
                    args.copy)
    rasterise_split(args.val_json, args.image_root,
                    args.out_root / "val_images",
                    args.out_root / "pseudo_labels_val",
                    args.copy)
    print(f"Done. Layout written under {args.out_root.resolve()}")


if __name__ == "__main__":
    main()
