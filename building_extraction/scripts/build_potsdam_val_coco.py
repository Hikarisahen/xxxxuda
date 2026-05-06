#!/usr/bin/env python3
"""Sample a Potsdam validation set from existing mask tiles, emit COCO JSON.

Why this exists: self-training rounds (R1/R2) currently validate on Vaihingen,
which means `best.pt` is selected on the *source* domain — actively biased
against transfer. We need a target-domain val set. UDA assumption is preserved:
these labels are used only for evaluation/model selection, never in training
loss.

Mask format auto-detection (override with --mask-format):
  * "rgb"     : 3-channel RGB, building = pure red (255, 0, 0)  [Vaihingen-style]
  * "binary"  : single-channel or RGB with grayscale, building = 255 (or > 127)
  * "classid" : single-channel label map, building = --building-class-id

Output:
  * COCO JSON with per-instance polygons (cv2.findContours)
  * A summary .txt next to it
  * The val_image_dir can point directly at the original IRRG dir; the
    CocoBuildingDataset only loads images named in the JSON.

CLI:
    python scripts/build_potsdam_val_coco.py \
        --mask-dir   /home/zfx/datasets/Potsdam/Masks_512 \
        --image-dir  /home/zfx/datasets/Potsdam/IRRG_512 \
        --output     annotations/potsdam_val_coco.json \
        --n-samples  150 \
        --seed       42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mask-dir", required=True, type=Path)
    p.add_argument("--image-dir", required=True, type=Path,
                   help="IRRG (or RGB) tile dir; filenames must match masks "
                        "(modulo extension and an optional _mask suffix).")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the COCO JSON.")
    p.add_argument("--n-samples", type=int, default=150,
                   help="How many tiles to sample. 150 is enough for a stable "
                        "AP estimate without burning labelling/eval time.")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed. Keep fixed so val set is reproducible.")
    p.add_argument("--mask-format", choices=["auto", "rgb", "binary", "classid"],
                   default="auto")
    p.add_argument("--building-class-id", type=int, default=1,
                   help="Only used when --mask-format=classid.")
    p.add_argument("--min-area", type=int, default=100,
                   help="Drop instances smaller than this many pixels.")
    p.add_argument("--max-area", type=int, default=500000,
                   help="Upper sanity cap; Potsdam buildings rarely exceed this.")
    p.add_argument("--epsilon-factor", type=float, default=0.002,
                   help="Polygon simplification factor for cv2.approxPolyDP.")
    p.add_argument("--require-image", action="store_true", default=True,
                   help="Skip masks whose paired image isn't found.")
    p.add_argument("--keep-empty", action="store_true",
                   help="Include tiles with zero buildings (useful for FP "
                        "rate). Default: drop them so val_loss isn't dominated "
                        "by trivial tiles.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Mask format detection + decoding
# ---------------------------------------------------------------------------
def detect_mask_format(sample_paths: List[Path]) -> str:
    """Peek at a few masks to guess the format.

    Heuristics:
      * 3-channel and contains (255, 0, 0) pixels -> "rgb"
      * single-channel or 3-channel-grayscale with values in {0, 255} -> "binary"
      * single-channel with a small set of small integer values -> "classid"
    """
    for p in sample_paths[:5]:
        arr = np.array(Image.open(p))
        if arr.ndim == 3 and arr.shape[2] >= 3:
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            if ((r == 255) & (g == 0) & (b == 0)).any():
                return "rgb"
            # Grayscale stored as RGB?
            if np.array_equal(r, g) and np.array_equal(g, b):
                arr = r
        if arr.ndim == 2:
            uniq = np.unique(arr)
            if set(uniq.tolist()).issubset({0, 255}):
                return "binary"
            if len(uniq) <= 10 and uniq.max() <= 20:
                return "classid"
    raise RuntimeError(
        "Could not auto-detect mask format. Pass --mask-format explicitly."
    )


def decode_building_mask(mask_arr: np.ndarray, fmt: str,
                         building_class_id: int) -> np.ndarray:
    """Return a uint8 binary mask where 1 = building."""
    if fmt == "rgb":
        if mask_arr.ndim != 3:
            raise ValueError("rgb format expects 3-channel mask")
        r, g, b = mask_arr[..., 0], mask_arr[..., 1], mask_arr[..., 2]
        return ((r == 255) & (g == 0) & (b == 0)).astype(np.uint8)
    if fmt == "binary":
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]  # assume grayscale-in-RGB
        return (mask_arr > 127).astype(np.uint8)
    if fmt == "classid":
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        return (mask_arr == building_class_id).astype(np.uint8)
    raise ValueError(f"Unknown mask format: {fmt}")


# ---------------------------------------------------------------------------
# Polygon extraction (mirrors convert_vaihingen_to_coco.py conventions)
# ---------------------------------------------------------------------------
def mask_to_instances(binary: np.ndarray, min_area: int, max_area: int,
                      epsilon_factor: float) -> List[dict]:
    instances: List[dict] = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) < 3:
                continue
            perim = cv2.arcLength(c, True)
            if perim < 1:
                continue
            approx = cv2.approxPolyDP(c, epsilon_factor * perim, True)
            poly = approx.flatten().tolist()
            if len(poly) < 6:
                continue
            xs, ys = poly[0::2], poly[1::2]
            x_min, y_min = min(xs), min(ys)
            w = max(xs) - x_min
            h = max(ys) - y_min
            if w <= 0 or h <= 0:
                continue
            instances.append({
                "polygon": poly,
                "area": float(area),
                "bbox": [float(x_min), float(y_min), float(w), float(h)],
            })
    return instances


# ---------------------------------------------------------------------------
# Image / mask pairing
# ---------------------------------------------------------------------------
IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def find_image(image_dir: Path, mask_path: Path) -> Path | None:
    """Try a few naming conventions to find the image paired with `mask_path`.
    Returns None if nothing matches.
    """
    stem = mask_path.stem
    candidates = [stem]
    if stem.endswith("_mask"):
        candidates.append(stem[:-5])
    if stem.endswith("_label"):
        candidates.append(stem[:-6])
    for s in candidates:
        for ext in IMG_EXTS:
            cand = image_dir / f"{s}{ext}"
            if cand.exists():
                return cand
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(
        [p for ext in IMG_EXTS for p in args.mask_dir.glob(f"*{ext}")]
    )
    if not mask_paths:
        raise SystemExit(f"No masks found in {args.mask_dir}")
    print(f"Found {len(mask_paths)} masks in {args.mask_dir}")

    fmt = args.mask_format
    if fmt == "auto":
        fmt = detect_mask_format(mask_paths)
        print(f"Auto-detected mask format: {fmt}")

    # Pair masks with images first; only keep masks with an existing image.
    pairs: List[Tuple[Path, Path]] = []
    n_orphan = 0
    for mp in mask_paths:
        ip = find_image(args.image_dir, mp)
        if ip is None:
            n_orphan += 1
            continue
        pairs.append((mp, ip))
    if n_orphan:
        print(f"  {n_orphan} masks had no matching image -> dropped")
    if not pairs:
        raise SystemExit("No mask/image pairs found.")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    images_out: List[dict] = []
    annotations_out: List[dict] = []
    next_img_id = 1
    next_ann_id = 1
    n_kept = 0
    n_skipped_empty = 0
    inst_counts: List[int] = []

    pbar = tqdm(total=args.n_samples, desc="Sampling val tiles")
    for mp, ip in pairs:
        if n_kept >= args.n_samples:
            break

        mask_arr = np.array(Image.open(mp))
        binary = decode_building_mask(mask_arr, fmt, args.building_class_id)
        instances = mask_to_instances(
            binary, args.min_area, args.max_area, args.epsilon_factor,
        )
        if not instances and not args.keep_empty:
            n_skipped_empty += 1
            continue

        with Image.open(ip) as im:
            W, H = im.size

        img_id = next_img_id; next_img_id += 1
        images_out.append({
            "id": img_id,
            "file_name": ip.name,
            "width": W,
            "height": H,
        })
        for inst in instances:
            annotations_out.append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": 0,
                "segmentation": [inst["polygon"]],
                "area": inst["area"],
                "bbox": inst["bbox"],
                "iscrowd": 0,
            })
            next_ann_id += 1
        inst_counts.append(len(instances))
        n_kept += 1
        pbar.update(1)
    pbar.close()

    if n_kept < args.n_samples:
        print(f"WARNING: only kept {n_kept} tiles (wanted {args.n_samples}). "
              f"{n_skipped_empty} tiles had no buildings and were skipped. "
              f"Pass --keep-empty if you want them.")

    coco = {
        "images": images_out,
        "annotations": annotations_out,
        "categories": [{"id": 0, "name": "building", "supercategory": "structure"}],
    }
    with open(args.output, "w") as f:
        json.dump(coco, f)

    ipt = np.array(inst_counts) if inst_counts else np.zeros(1)
    summary = [
        f"Mask dir       : {args.mask_dir}",
        f"Image dir      : {args.image_dir}",
        f"Mask format    : {fmt}",
        f"Sampling seed  : {args.seed}",
        f"Tiles requested: {args.n_samples}",
        f"Tiles kept     : {n_kept}",
        f"Tiles skipped (empty): {n_skipped_empty}",
        f"Orphan masks (no image): {n_orphan}",
        f"Total instances: {len(annotations_out)}",
        f"Per-tile instance count: mean={ipt.mean():.2f}  median={np.median(ipt):.0f}  "
        f"min={ipt.min()}  max={ipt.max()}",
    ]
    summary_path = args.output.with_suffix("").as_posix() + "_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary))
    print("\n" + "\n".join(summary))
    print(f"\nWrote val COCO: {args.output}")
    print(f"Summary       : {summary_path}")


if __name__ == "__main__":
    main()
