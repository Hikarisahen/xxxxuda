#!/usr/bin/env python3
"""Generate high-confidence pseudo-labels on a target-domain image directory
using a trained SMP-UNet source model.

Steps per tile:
  1. forward -> sigmoid probability map
  2. binarise at a HIGH threshold (default 0.85) so we only keep what the
     source model is genuinely confident about
  3. morphological open + connected-component area filter to drop hairline
     halo and isolated noise
  4. drop the whole tile if it ends up too sparse (no useful supervision)

Output layout (matches what configs/self_training_round*.yaml expects):
    <out-image-dir>/<stem>.<ext>          (symlink to the source tile)
    <out-mask-dir>/<stem>_mask.png        (binary 0/255)
    <out-mask-dir>/_summary.txt           (per-tile stats + global aggregates)
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.building_model import build_building_model


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img: np.ndarray, size: int) -> torch.Tensor:
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def cleanup_mask(mask: np.ndarray, open_k: int, min_area: int) -> np.ndarray:
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if min_area > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        mask = out
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Source-model config (for arch/encoder).")
    ap.add_argument("--checkpoint", required=True, help="Source-model checkpoint.")
    ap.add_argument("--image-dir", required=True, type=Path, help="Target-domain tiles.")
    ap.add_argument("--out-image-dir", required=True, type=Path,
                    help="Where to symlink the kept tiles (those that have useful pseudo-labels).")
    ap.add_argument("--out-mask-dir", required=True, type=Path,
                    help="Where to write <stem>_mask.png pseudo-labels.")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="Confidence threshold for binarisation. Higher = cleaner labels, fewer kept tiles.")
    ap.add_argument("--min-pos-ratio", type=float, default=0.02,
                    help="Drop a tile if final pos-pixel ratio is below this (no useful supervision).")
    ap.add_argument("--max-pos-ratio", type=float, default=0.85,
                    help="Drop a tile if final pos-pixel ratio is above this "
                         "(probably hallucinated — collapse to all-positive).")
    ap.add_argument("--open-kernel", type=int, default=3, help="Morphological open kernel size; 0 to disable.")
    ap.add_argument("--min-blob-area", type=int, default=300,
                    help="Drop connected components smaller than this many pixels.")
    ap.add_argument("--infer-size", type=int, default=512)
    ap.add_argument("--copy-images", action="store_true", help="Copy instead of symlink.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_image_dir.mkdir(parents=True, exist_ok=True)
    args.out_mask_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = build_building_model(
        arch=cfg["model"].get("arch", "smp_unet"),
        encoder_name=cfg["model"].get("encoder_name", "tu-convnext_base"),
        encoder_weights=None,
        num_classes=cfg["model"].get("num_classes", 1),
    )
    sd = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(args.device).eval()

    paths = sorted(
        list(args.image_dir.glob("*.tif")) + list(args.image_dir.glob("*.tiff"))
        + list(args.image_dir.glob("*.png")) + list(args.image_dir.glob("*.jpg"))
    )
    print(f"Processing {len(paths)} tiles from {args.image_dir}")

    n_kept = n_skipped_low = n_skipped_high = 0
    pos_ratios = []

    with torch.no_grad():
        for ip in tqdm(paths, desc="Pseudo-label"):
            img = np.array(Image.open(ip).convert("RGB"))
            x = preprocess(img, args.infer_size).to(args.device)
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            if prob.shape != img.shape[:2]:
                prob = cv2.resize(prob, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
            mask = (prob > args.threshold).astype(np.uint8) * 255
            mask = cleanup_mask(mask, args.open_kernel, args.min_blob_area)

            ratio = (mask > 0).mean()
            if ratio < args.min_pos_ratio:
                n_skipped_low += 1
                continue
            if ratio > args.max_pos_ratio:
                n_skipped_high += 1
                continue

            out_img_path = args.out_image_dir / ip.name
            if not out_img_path.exists():
                if args.copy_images:
                    shutil.copy2(ip, out_img_path)
                else:
                    try:
                        out_img_path.symlink_to(ip.resolve())
                    except OSError:
                        shutil.copy2(ip, out_img_path)
            Image.fromarray(mask).save(args.out_mask_dir / f"{ip.stem}_mask.png")

            pos_ratios.append(ratio)
            n_kept += 1

    pr = np.array(pos_ratios) if pos_ratios else np.zeros(1)
    summary = [
        f"Source ckpt   : {args.checkpoint}",
        f"Image dir     : {args.image_dir}",
        f"Threshold     : {args.threshold}",
        f"Open kernel   : {args.open_kernel}",
        f"Min blob area : {args.min_blob_area}",
        f"Pos-ratio range allowed: [{args.min_pos_ratio}, {args.max_pos_ratio}]",
        "",
        f"Total tiles processed : {len(paths)}",
        f"Kept (good pseudo-lbl): {n_kept}",
        f"Skipped (too sparse)  : {n_skipped_low}",
        f"Skipped (too dense)   : {n_skipped_high}",
        "",
        f"Per-kept-tile pos ratio: mean={pr.mean():.3f}  median={np.median(pr):.3f}  "
        f"min={pr.min():.3f}  max={pr.max():.3f}",
    ]
    summary_str = "\n".join(summary)
    (args.out_mask_dir / "_summary.txt").write_text(summary_str)
    print("\n" + summary_str)


if __name__ == "__main__":
    main()
