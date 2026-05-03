#!/usr/bin/env python3
"""Per-instance visualisation of an mmdet Mask R-CNN model.

Layout per saved grid (4 panels):
    [original | per-instance coloured masks | overlay (img + masks + bboxes) | GT contours]

The 4th panel is only drawn when --mask-dir is given (i.e. you have ground
truth or pseudo-labels for the same tile). For unlabelled target-domain
images, omit --mask-dir and the script saves a 3-panel grid.

Each instance gets a deterministic random colour seeded from its index in the
prediction list, so the same tile renders the same way across runs.

CLI examples:

  # Vaihingen val (with GT contours)
  python scripts/visualize_instances.py \
      --config configs/mmdet/mask_rcnn_vaihingen.py \
      --checkpoint checkpoints/mmdet_vaihingen/best_segm_mAP.pth \
      --image-dir data/val_images \
      --mask-dir  data/pseudo_labels_val \
      --out-dir   visualisations_mmdet_vaihingen --limit 20

  # Potsdam (no GT)
  python scripts/visualize_instances.py \
      --config configs/mmdet/mask_rcnn_vaihingen.py \
      --checkpoint checkpoints/mmdet_vaihingen/best_segm_mAP.pth \
      --image-dir /home/zfx/datasets/Potsdam/IRRG_512 \
      --out-dir   visualisations_mmdet_potsdam_baseline --limit 30 --seed 0
"""
import argparse
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image-dir", required=True, type=Path)
    p.add_argument("--mask-dir", type=Path, default=None,
                   help="Optional: directory of <stem>_mask.png GT/pseudo-label "
                        "binary masks. If given, GT contours are drawn in cyan.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0,
                   help="Sampling seed so the same N images are picked across runs.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def color_for_index(i: int) -> tuple:
    """Deterministic distinct-ish colour per instance."""
    rng = random.Random(int(i) * 9973 + 17)
    # Bias away from pure white/black by clamping each channel to [60, 240].
    return (rng.randint(60, 240), rng.randint(60, 240), rng.randint(60, 240))


def render_instances(img: np.ndarray, masks: np.ndarray, bboxes: np.ndarray,
                     scores: np.ndarray, draw_bboxes: bool) -> tuple:
    """Return (instance_panel, overlay_panel) — both H x W x 3 uint8.

    instance_panel: black canvas, each mask filled with its colour.
    overlay_panel : original image with masks blended + thin contour + bboxes.
    """
    H, W = img.shape[:2]
    inst = np.zeros_like(img)
    overlay = img.copy()

    for i in range(len(masks)):
        m = masks[i].astype(np.uint8)
        if m.sum() == 0:
            continue
        col = color_for_index(i)
        col_arr = np.array(col, dtype=np.uint8)

        # Solid colour panel
        inst[m.astype(bool)] = col_arr

        # Blended overlay (50/50)
        sel = m.astype(bool)
        overlay[sel] = (0.5 * overlay[sel] + 0.5 * col_arr).astype(np.uint8)

        # Crisp contour line on the overlay
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, col, thickness=2)

        if draw_bboxes:
            x1, y1, x2, y2 = bboxes[i].astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 1)
            label = f"{scores[i]:.2f}"
            cv2.putText(overlay, label, (x1 + 2, max(12, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    return inst, overlay


def gt_contour_panel(img: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    if gt_mask is None or gt_mask.sum() == 0:
        return out
    contours, _ = cv2.findContours(gt_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), thickness=2)  # cyan
    return out


def load_gt_mask(mask_dir: Optional[Path], stem: str, shape: tuple) -> Optional[np.ndarray]:
    if mask_dir is None:
        return None
    for ext in (".png", ".tif", ".tiff", ".jpg"):
        p = mask_dir / f"{stem}_mask{ext}"
        if p.exists():
            m = np.array(Image.open(p).convert("L"))
            if m.shape != shape:
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            return (m > 127).astype(np.uint8)
    return None


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from mmdet.apis import init_detector, inference_detector

    print(f"Loading model: {args.checkpoint}")
    model = init_detector(args.config, args.checkpoint, device=args.device)

    paths = sorted(
        list(args.image_dir.glob("*.tif")) + list(args.image_dir.glob("*.tiff"))
        + list(args.image_dir.glob("*.png")) + list(args.image_dir.glob("*.jpg"))
    )
    print(f"Found {len(paths)} images under {args.image_dir}")

    rng = random.Random(args.seed)
    if len(paths) > args.limit:
        paths = rng.sample(paths, args.limit)

    n_inst_per_img: List[int] = []
    score_means: List[float] = []

    for ip in tqdm(paths, desc="Visualise"):
        img = np.array(Image.open(ip).convert("RGB"))
        H, W = img.shape[:2]

        result = inference_detector(model, img)
        pred = result.pred_instances
        keep = pred.scores >= args.score_thresh
        scores = pred.scores[keep].cpu().numpy()
        labels = pred.labels[keep].cpu().numpy()
        bboxes = pred.bboxes[keep].cpu().numpy()
        masks = pred.masks[keep].cpu().numpy()

        # Restrict to building class
        bld = labels == 0
        scores, bboxes, masks = scores[bld], bboxes[bld], masks[bld]

        n_inst_per_img.append(len(scores))
        if len(scores):
            score_means.append(float(scores.mean()))

        inst_panel, overlay_panel = render_instances(img, masks, bboxes, scores,
                                                     draw_bboxes=True)

        gt_mask = load_gt_mask(args.mask_dir, ip.stem, (H, W))
        panels = [img, inst_panel, overlay_panel]
        if gt_mask is not None:
            panels.append(gt_contour_panel(img, gt_mask))

        # Resize defensively (all are H x W already, but keep_ratio false-positive)
        panels = [cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) for p in panels]
        grid = np.concatenate(panels, axis=1)
        Image.fromarray(grid).save(args.out_dir / f"{ip.stem}_grid.png")

    n = np.array(n_inst_per_img)
    sm = np.array(score_means) if score_means else np.zeros(1)
    summary = [
        f"Images visualised : {len(paths)}",
        f"Score threshold   : {args.score_thresh}",
        "",
        f"Instances per image : mean={n.mean():.2f}  median={np.median(n):.0f}  "
        f"min={n.min()}  max={n.max()}",
        f"Mean score (per-img): mean={sm.mean():.3f}  min={sm.min():.3f}  max={sm.max():.3f}",
        "",
        "How to read this:",
        "  * Panel 2 (instance colours) — touching buildings should get DIFFERENT colours.",
        "    If two adjacent buildings share one colour, instance separation is failing.",
        "  * Panel 3 (overlay)         — coloured contours track the actual rooflines.",
        "    Halo / spillover into roads = false-positive masks.",
        "  * Panel 4 (GT cyan)         — only present when --mask-dir was supplied.",
        "    Each cyan line should be hugged by exactly one coloured contour.",
    ]
    (args.out_dir / "summary.txt").write_text("\n".join(summary))
    print("\n".join(summary))
    print(f"\nGrids + summary in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
