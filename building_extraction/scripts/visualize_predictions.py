#!/usr/bin/env python3
"""Visualise + sanity-check a trained building segmentation model.

Outputs (under --out-dir):
  * grid_<idx>.png  -- 4-panel: image | GT | pred-prob | overlay (GT cyan, pred magenta)
  * summary.txt     -- per-image IoU/Dice + global aggregates, plus a separate
                       aggregate over ONLY non-empty GT tiles (the honest number).

Why a separate "non-empty" aggregate:
  Val tiles with no buildings give Dice=1.0 by the smooth-term trick when the
  model also predicts empty. Including them in the average inflates the score
  past anything meaningful. The "non-empty" aggregate tells you how good the
  model actually is at the buildings task.
"""
import argparse
import os
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


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L"))
    return (m > 127).astype(np.uint8)


def preprocess(img: np.ndarray, size: int) -> torch.Tensor:
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Image + GT (cyan) + pred (magenta), with intersection as white."""
    out = img.copy().astype(np.float32)
    if gt.shape != out.shape[:2]:
        gt = cv2.resize(gt, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    if pred.shape != out.shape[:2]:
        pred = cv2.resize(pred, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)

    gt_layer = np.zeros_like(out); gt_layer[..., 1] = 255; gt_layer[..., 2] = 255  # cyan
    pr_layer = np.zeros_like(out); pr_layer[..., 0] = 255; pr_layer[..., 2] = 255  # magenta

    out = np.where(gt[..., None].astype(bool), 0.5 * out + 0.5 * gt_layer, out)
    out = np.where(pred[..., None].astype(bool), 0.5 * out + 0.5 * pr_layer, out)
    return out.clip(0, 255).astype(np.uint8)


def make_grid(img: np.ndarray, gt: np.ndarray, prob: np.ndarray,
              pred_bin: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    gt_vis = (gt * 255).astype(np.uint8)
    gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2RGB)
    prob_vis = (prob * 255).astype(np.uint8)
    prob_vis = cv2.applyColorMap(prob_vis, cv2.COLORMAP_VIRIDIS)
    prob_vis = cv2.cvtColor(prob_vis, cv2.COLOR_BGR2RGB)
    ovl = overlay(img, gt, pred_bin)

    # Resize all to h,w just in case
    panels = [img, gt_vis, prob_vis, ovl]
    panels = [cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in panels]
    return np.concatenate(panels, axis=1)


def dice_iou(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    psum, gsum = p.sum(), g.sum()
    dice = (2 * inter + smooth) / (psum + gsum + smooth)
    iou = (inter + smooth) / (psum + gsum - inter + smooth)
    return float(dice), float(iou), int(gsum), int(psum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/unet_finetune.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/unet/best_unet.pt")
    ap.add_argument("--image-dir", default=None,
                    help="Defaults to data.val_images from config.")
    ap.add_argument("--mask-dir", default=None,
                    help="Defaults to data.val_masks from config.")
    ap.add_argument("--out-dir", default="visualisations")
    ap.add_argument("--limit", type=int, default=20,
                    help="Save the first N visualisation grids.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    image_dir = Path(args.image_dir or cfg["data"]["val_images"])
    mask_dir = Path(args.mask_dir or cfg["data"]["val_masks"])
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    img_size = cfg["data"]["image_size"]

    # Build model + load checkpoint
    model = build_building_model(
        arch=cfg["model"].get("arch", "smp_unet"),
        encoder_name=cfg["model"].get("encoder_name", "tu-convnext_base"),
        encoder_weights=None,                  # we're loading our own ckpt
        num_classes=cfg["model"].get("num_classes", 1),
        dino_model_name=cfg["model"].get("dino_model_name", "dinov2_vitb14"),
        dino_freeze=cfg["model"].get("dino_freeze", True),
    )
    sd = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[ckpt] partial load: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(args.device).eval()

    img_paths = sorted(
        list(image_dir.glob("*.tif")) + list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.jpg"))
    )
    print(f"Found {len(img_paths)} val images.")

    per_img = []
    saved = 0
    with torch.no_grad():
        for i, ip in enumerate(tqdm(img_paths, desc="Predict")):
            img = load_image(ip)
            mp = mask_dir / f"{ip.stem}_mask.png"
            gt = load_mask(mp) if mp.exists() else np.zeros(img.shape[:2], np.uint8)

            x = preprocess(img, img_size).to(args.device)
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            # Resize prob back to image native size for fair comparison
            if prob.shape != gt.shape:
                prob = cv2.resize(prob, (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
            pred_bin = (prob > args.threshold).astype(np.uint8)

            dice, iou, gsum, psum = dice_iou(pred_bin, gt)
            per_img.append({
                "name": ip.name, "dice": dice, "iou": iou,
                "gt_pixels": gsum, "pred_pixels": psum,
            })

            if saved < args.limit:
                grid = make_grid(img, gt, prob, pred_bin)
                Image.fromarray(grid).save(out_dir / f"grid_{i:04d}_{ip.stem}.png")
                saved += 1

    # Aggregates
    all_dice = np.array([r["dice"] for r in per_img])
    all_iou = np.array([r["iou"] for r in per_img])
    nonempty = np.array([r["gt_pixels"] > 0 for r in per_img])
    empty = ~nonempty

    lines = []
    lines.append(f"Total val images: {len(per_img)}")
    lines.append(f"  with buildings (non-empty GT): {int(nonempty.sum())}")
    lines.append(f"  empty GT (no buildings):       {int(empty.sum())}")
    lines.append("")
    lines.append("---- ALL images (includes trivially-perfect empty ones) ----")
    lines.append(f"  mean Dice: {all_dice.mean():.4f}")
    lines.append(f"  mean IoU : {all_iou.mean():.4f}")
    lines.append("")
    lines.append("---- Non-empty GT only (the honest number) ----")
    if nonempty.any():
        lines.append(f"  mean Dice: {all_dice[nonempty].mean():.4f}")
        lines.append(f"  mean IoU : {all_iou[nonempty].mean():.4f}")
    else:
        lines.append("  (no non-empty GT tiles found — check mask paths!)")
    lines.append("")
    lines.append("---- Bottom-10 by IoU (worst predictions) ----")
    worst = sorted(per_img, key=lambda r: r["iou"])[:10]
    for r in worst:
        lines.append(f"  IoU={r['iou']:.3f} Dice={r['dice']:.3f} "
                     f"gt_px={r['gt_pixels']:>7} pred_px={r['pred_pixels']:>7}  {r['name']}")

    out_txt = out_dir / "summary.txt"
    out_txt.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nGrids + summary saved under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
