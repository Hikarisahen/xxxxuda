#!/usr/bin/env python3
"""Run a trained building model on unlabeled target-domain images and save
side-by-side visualisations.

Layout per saved grid:  [original | prob heatmap | binary pred | overlay]

Use this to eyeball the source-domain → target-domain gap before any self-
training, and to inspect each self-training iteration's output.
"""
import argparse
import random
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


def preprocess(img: np.ndarray, infer_size: int) -> torch.Tensor:
    """Resize to model's expected size, normalise, return [1,3,H,W]."""
    if img.shape[0] != infer_size or img.shape[1] != infer_size:
        img = cv2.resize(img, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def make_overlay(img: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Image with predicted-building region tinted magenta."""
    out = img.copy().astype(np.float32)
    if pred.shape != out.shape[:2]:
        pred = cv2.resize(pred, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    layer = np.zeros_like(out); layer[..., 0] = 255; layer[..., 2] = 255  # magenta
    out = np.where(pred[..., None].astype(bool), 0.5 * out + 0.5 * layer, out)
    return out.clip(0, 255).astype(np.uint8)


def make_grid(img: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    prob_vis = (prob * 255).astype(np.uint8)
    prob_vis = cv2.applyColorMap(prob_vis, cv2.COLORMAP_VIRIDIS)
    prob_vis = cv2.cvtColor(prob_vis, cv2.COLOR_BGR2RGB)

    pred_vis = (pred * 255).astype(np.uint8)
    pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2RGB)

    ovl = make_overlay(img, pred)
    panels = [img, prob_vis, pred_vis, ovl]
    panels = [cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in panels]
    return np.concatenate(panels, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/unet_finetune.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/unet/best_unet.pt")
    ap.add_argument("--image-dir", required=True, help="Target-domain images directory.")
    ap.add_argument("--out-dir", default="visualisations_target")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--infer-size", type=int, default=512,
                    help="Resize input to this square before inference (matches source training).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Sampling seed so the same N images are picked across runs.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model = build_building_model(
        arch=cfg["model"].get("arch", "smp_unet"),
        encoder_name=cfg["model"].get("encoder_name", "tu-convnext_base"),
        encoder_weights=None,
        num_classes=cfg["model"].get("num_classes", 1),
        dino_model_name=cfg["model"].get("dino_model_name", "dinov2_vitb14"),
        dino_freeze=cfg["model"].get("dino_freeze", True),
    )
    sd = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(args.device).eval()

    image_dir = Path(args.image_dir)
    paths = sorted(
        list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff"))
    )
    print(f"Found {len(paths)} target-domain images.")

    rng = random.Random(args.seed)
    if len(paths) > args.limit:
        paths = rng.sample(paths, args.limit)

    # Aggregate diagnostics — useful even without GT.
    prob_means = []
    pred_pos_ratios = []

    with torch.no_grad():
        for ip in tqdm(paths, desc="Infer"):
            img = np.array(Image.open(ip).convert("RGB"))
            x = preprocess(img, args.infer_size).to(args.device)
            logits = model(x)
            prob_low = torch.sigmoid(logits)[0, 0].cpu().numpy()
            # Resize prob back to native image size for overlay quality
            prob = cv2.resize(prob_low, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            pred = (prob > args.threshold).astype(np.uint8)

            prob_means.append(float(prob.mean()))
            pred_pos_ratios.append(float(pred.mean()))

            grid = make_grid(img, prob, pred)
            Image.fromarray(grid).save(out_dir / f"{ip.stem}_grid.png")

    pm = np.array(prob_means); pr = np.array(pred_pos_ratios)
    summary = [
        f"Images visualised: {len(paths)}",
        f"Inference resolution: {args.infer_size}x{args.infer_size}",
        f"Threshold: {args.threshold}",
        "",
        f"prob_mean  (per-image, mean ± std)   : {pm.mean():.3f} ± {pm.std():.3f}",
        f"  range  [min, max]                 : [{pm.min():.3f}, {pm.max():.3f}]",
        f"pred_pos_ratio (per-image, mean ± std): {pr.mean():.3f} ± {pr.std():.3f}",
        f"  range  [min, max]                 : [{pr.min():.3f}, {pr.max():.3f}]",
        "",
        "How to read this:",
        "  * Healthy target-domain inference looks like Vaihingen val: ",
        "    prob_mean ~ pred_pos_ratio, both close to the actual building ratio of the scene.",
        "  * Collapse to ~0   -> source model failing on target distribution (expected).",
        "  * Collapse to ~1   -> over-prediction, probably hallucinating buildings everywhere.",
        "  * High variance    -> model works on some tiles, fails on others (typical mid-gap).",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary))
    print("\n".join(summary))
    print(f"\nGrids + summary in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
