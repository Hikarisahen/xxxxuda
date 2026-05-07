#!/usr/bin/env python3
"""Compute per-channel mean/std over one or more image directories.

Why we need this: torchvision's MaskRCNN normalises inputs with ImageNet RGB
defaults (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Our inputs
are IRRG (R=NIR, G=visible-red, B=visible-green) — channel statistics are
significantly different, so the backbone sees a distribution shift relative
to its pretraining. Replacing the normalisation constants with values
computed on the actual data closes that gap.

Recommendation:
  * Compute joint statistics over Vaihingen train + Potsdam unlabeled images.
    This minimises the train-test shift in the *normalised input* space,
    which is what the backbone actually sees.
  * Channels are reported in the order they appear in the file (for IRRG
    GeoTIFF this is typically NIR, R, G).

CLI:
    python scripts/compute_channel_stats.py \
        --image-dirs /home/zfx/datasets/Vaihingen_croped/train/Images \
                     /home/zfx/datasets/Potsdam/IRRG_512 \
        --max-per-dir 1500 \
        --output     configs/channel_stats.yaml

Plug the result into your config under model.image_mean / model.image_std
(see updated build_model in src/training/train_maskrcnn.py).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dirs", nargs="+", required=True, type=Path,
                   help="One or more dirs to scan. Stats are reported "
                        "per-dir and combined.")
    p.add_argument("--max-per-dir", type=int, default=1500,
                   help="Cap images sampled from each dir (random subsample) "
                        "for speed; 1500 gives stable stats to 3 decimal places.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None,
                   help="Optional YAML output path.")
    return p.parse_args()


def list_images(d: Path) -> List[Path]:
    return sorted([p for ext in IMG_EXTS for p in d.glob(f"*{ext}")])


def accumulate(paths: List[Path], n_channels: int):
    """Single-pass per-channel sum / sum-of-squares accumulator over [0,1] floats."""
    sum_c = np.zeros(n_channels, dtype=np.float64)
    sumsq_c = np.zeros(n_channels, dtype=np.float64)
    n_pixels = 0
    for p in tqdm(paths, desc=f"  {p.parent.name if paths else ''}", leave=False):
        arr = np.array(Image.open(p))
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = arr.astype(np.float64) / 255.0
        if arr.shape[2] != n_channels:
            # First image determined n_channels; if a later image differs, skip.
            continue
        flat = arr.reshape(-1, n_channels)
        sum_c += flat.sum(axis=0)
        sumsq_c += (flat ** 2).sum(axis=0)
        n_pixels += flat.shape[0]
    return sum_c, sumsq_c, n_pixels


def stats_from_accum(sum_c, sumsq_c, n_pixels):
    mean = sum_c / max(1, n_pixels)
    var = (sumsq_c / max(1, n_pixels)) - mean ** 2
    var = np.clip(var, 0.0, None)
    std = np.sqrt(var)
    return mean.tolist(), std.tolist()


def fmt(v: List[float]) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # First image dictates channel count.
    n_channels = None
    per_dir_paths = []
    for d in args.image_dirs:
        if not d.exists():
            raise SystemExit(f"Directory not found: {d}")
        all_paths = list_images(d)
        if not all_paths:
            print(f"WARNING: no images in {d}")
            per_dir_paths.append([])
            continue
        rng.shuffle(all_paths)
        sample = all_paths[: args.max_per_dir]
        per_dir_paths.append(sample)
        if n_channels is None:
            probe = np.array(Image.open(sample[0]))
            n_channels = 1 if probe.ndim == 2 else probe.shape[2]
            print(f"Detected {n_channels} channels from {sample[0].name}")

    # Accumulate per-dir + combined.
    print("\n--- Per-directory statistics ---")
    combined_sum = np.zeros(n_channels, dtype=np.float64)
    combined_sumsq = np.zeros(n_channels, dtype=np.float64)
    combined_n = 0
    per_dir_results = {}

    for d, paths in zip(args.image_dirs, per_dir_paths):
        if not paths:
            continue
        print(f"\n{d}  ({len(paths)} images)")
        s, sq, n = accumulate(paths, n_channels)
        mean, std = stats_from_accum(s, sq, n)
        print(f"  mean: {fmt(mean)}")
        print(f"  std : {fmt(std)}")
        per_dir_results[str(d)] = {"mean": mean, "std": std, "n_images": len(paths)}
        combined_sum += s
        combined_sumsq += sq
        combined_n += n

    print("\n--- Combined (recommended for normalisation) ---")
    if combined_n > 0:
        c_mean, c_std = stats_from_accum(combined_sum, combined_sumsq, combined_n)
        print(f"  mean: {fmt(c_mean)}")
        print(f"  std : {fmt(c_std)}")
    else:
        c_mean, c_std = [0.0] * n_channels, [1.0] * n_channels
        print("  (no images accumulated — falling back to identity normalisation)")

    print("\n--- For pasting into config (under model:) ---")
    print(f"  image_mean: {fmt(c_mean)}")
    print(f"  image_std:  {fmt(c_std)}")

    out = {
        "n_channels": n_channels,
        "combined": {"mean": c_mean, "std": c_std},
        "per_directory": per_dir_results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False)
        print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
