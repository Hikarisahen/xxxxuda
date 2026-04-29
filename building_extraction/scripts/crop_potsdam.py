#!/usr/bin/env python3
"""Crop Potsdam orthophotos (and optional labels) into 512x512 tiles.

Matches the Vaihingen tile naming convention so the rest of the pipeline
(self-training, visualize_unlabeled) can pick them up unchanged:

    out_image_dir/<orig_stem>_<row>_<col>.tif         (RGB-mode 3-channel)
    out_label_dir/<orig_stem>_<row>_<col>_mask.png    (binary 0/255)

Building label conversion (Potsdam class scheme):
    Class            (R,   G,   B)
    impervious       (255, 255, 255)
    building         (0,   0,   255)   <-- this becomes 255 in our binary mask
    low_veg          (0,   255, 255)
    tree             (0,   255, 0)
    car              (255, 255, 0)
    clutter          (255, 0,   0)
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


BUILDING_RGB = (0, 0, 255)


def crop_image(img: np.ndarray, tile: int, stride: int):
    H, W = img.shape[:2]
    for r in range(0, H - tile + 1, stride):
        for c in range(0, W - tile + 1, stride):
            yield r, c, img[r:r + tile, c:c + tile]


def label_to_building_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = BUILDING_RGB
    mask = ((rgb[..., 0] == r) & (rgb[..., 1] == g) & (rgb[..., 2] == b))
    return (mask * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path, required=True,
                    help="Directory with the unzipped Potsdam IRRG tiles.")
    ap.add_argument("--label-dir", type=Path, default=None,
                    help="Optional: dir with Labels_all_noBoundary RGB labels. "
                         "If given, building masks are produced.")
    ap.add_argument("--out-image-dir", type=Path, required=True)
    ap.add_argument("--out-label-dir", type=Path, default=None)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512,
                    help="Use 512 for non-overlapping tiles, smaller for overlap.")
    ap.add_argument("--image-suffix", default="_IRRG.tif")
    ap.add_argument("--label-suffix", default="_label_noBoundary.tif",
                    help="Tail of the Potsdam label filename. Auto-detect tries "
                         "common variants if this exact one isn't found.")
    args = ap.parse_args()

    args.out_image_dir.mkdir(parents=True, exist_ok=True)
    if args.label_dir and args.out_label_dir:
        args.out_label_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(args.image_dir.glob(f"*{args.image_suffix}"))
    if not img_paths:
        # Fallback: any .tif in the dir
        img_paths = sorted(args.image_dir.glob("*.tif"))
    print(f"Found {len(img_paths)} source orthophotos in {args.image_dir}")

    n_img_tiles = 0
    n_label_tiles = 0
    label_variants = [args.label_suffix, "_label.tif", ".tif"]

    for ip in tqdm(img_paths, desc="Cropping"):
        # Stem like 'top_potsdam_2_10' (strip the IRRG/RGB suffix).
        full_stem = ip.name
        for s in (args.image_suffix, ".tif"):
            if full_stem.endswith(s):
                stem = full_stem[: -len(s)]
                break
        else:
            stem = ip.stem

        img = np.array(Image.open(ip).convert("RGB"))

        label = None
        if args.label_dir:
            for sfx in label_variants:
                lp = args.label_dir / f"{stem}{sfx}"
                if lp.exists():
                    label = np.array(Image.open(lp).convert("RGB"))
                    break

        for r, c, tile in crop_image(img, args.tile, args.stride):
            tile_name = f"{stem}_{r}_{c}.tif"
            Image.fromarray(tile).save(args.out_image_dir / tile_name)
            n_img_tiles += 1

            if label is not None and args.out_label_dir:
                ltile = label[r:r + args.tile, c:c + args.tile]
                bmask = label_to_building_mask(ltile)
                mask_name = f"{stem}_{r}_{c}_mask.png"
                Image.fromarray(bmask).save(args.out_label_dir / mask_name)
                n_label_tiles += 1

    print(f"Done. {n_img_tiles} image tiles -> {args.out_image_dir}")
    if n_label_tiles:
        print(f"      {n_label_tiles} label tiles -> {args.out_label_dir}")


if __name__ == "__main__":
    main()
