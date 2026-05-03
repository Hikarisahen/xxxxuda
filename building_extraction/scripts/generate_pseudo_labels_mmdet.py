#!/usr/bin/env python3
"""Generate instance-level pseudo-labels on a target-domain image directory
using a trained mmdet Mask R-CNN model. Output is a single COCO JSON that
mmdet can consume directly as a training dataset.

Per-tile pipeline:
    1. inference_detector -> per-instance score, bbox, mask
    2. drop instances with score < --score-thresh (default 0.85, tight)
    3. per-instance area filter via cv2.contourArea
    4. per-instance bbox aspect-ratio filter
    5. drop the entire tile if no instance survives
    6. emit a COCO-format record (images[] + annotations[])

Tiles with surviving pseudo-labels are also symlinked into --out-image-dir so
mmdet's CocoDataset finds them via `data_prefix`.

CLI:
    python scripts/generate_pseudo_labels_mmdet.py \
        --config       configs/mmdet/mask_rcnn_vaihingen.py \
        --checkpoint   checkpoints/mmdet_vaihingen/best_segm_mAP.pth \
        --image-dir    /home/zfx/datasets/Potsdam/IRRG_512 \
        --out-image-dir data/potsdam_pseudo_images \
        --out-coco     data/potsdam_pseudo_coco.json \
        --score-thresh 0.85 \
        --min-area     300 \
        --aspect-ratio-range 0.2 5.0
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

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
    p.add_argument("--out-image-dir", required=True, type=Path,
                   help="Where to symlink (or copy) the kept tile images.")
    p.add_argument("--out-coco", required=True, type=Path,
                   help="Path to write the combined COCO pseudo-annotation JSON.")
    p.add_argument("--score-thresh", type=float, default=0.85)
    p.add_argument("--min-area", type=int, default=300)
    p.add_argument("--max-area", type=int, default=300000)
    p.add_argument("--aspect-ratio-range", type=float, nargs=2, default=(0.2, 5.0),
                   metavar=("MIN", "MAX"),
                   help="Drop instances whose bbox w/h is outside this range.")
    p.add_argument("--min-instances-per-tile", type=int, default=1,
                   help="Skip tiles that end up with fewer than this many kept instances.")
    p.add_argument("--copy-images", action="store_true",
                   help="Copy instead of symlink (use when symlinks aren't supported).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def filter_instances(pred, score_thresh: float, min_area: int, max_area: int,
                     ar_min: float, ar_max: float) -> List[Dict]:
    """Return list of records: [{polygon, area, bbox(XYWH), score}, ...]."""
    records: List[Dict] = []
    if len(pred) == 0:
        return records

    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()
    bboxes = pred.bboxes.cpu().numpy()
    masks = pred.masks.cpu().numpy()

    for i in range(len(scores)):
        if scores[i] < score_thresh or labels[i] != 0:
            continue
        m = masks[i].astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mmdet may return masks broken into multiple contours; keep all valid.
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue
            ar = w / h
            if ar < ar_min or ar > ar_max:
                continue
            polygon = c.flatten().tolist()
            if len(polygon) < 6:
                continue
            records.append({
                "polygon": polygon,
                "area": area,
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(scores[i]),
            })
    return records


def main():
    args = parse_args()
    args.out_image_dir.mkdir(parents=True, exist_ok=True)
    args.out_coco.parent.mkdir(parents=True, exist_ok=True)

    from mmdet.apis import init_detector, inference_detector

    print(f"Loading model: {args.checkpoint}")
    model = init_detector(args.config, args.checkpoint, device=args.device)

    paths = sorted(
        list(args.image_dir.glob("*.tif")) + list(args.image_dir.glob("*.tiff"))
        + list(args.image_dir.glob("*.png")) + list(args.image_dir.glob("*.jpg"))
    )
    print(f"Found {len(paths)} tiles")

    images_out: List[Dict] = []
    annotations_out: List[Dict] = []
    next_img_id = 1
    next_ann_id = 1

    n_kept = n_skipped = 0
    inst_per_kept_tile: List[int] = []
    score_per_kept_inst: List[float] = []

    ar_min, ar_max = args.aspect_ratio_range

    for ip in tqdm(paths, desc="Pseudo-label"):
        img = np.array(Image.open(ip).convert("RGB"))
        H, W = img.shape[:2]
        result = inference_detector(model, img)
        records = filter_instances(result.pred_instances,
                                   args.score_thresh, args.min_area, args.max_area,
                                   ar_min, ar_max)
        if len(records) < args.min_instances_per_tile:
            n_skipped += 1
            continue

        # Symlink (or copy) the tile so mmdet can find it under one prefix.
        out_img = args.out_image_dir / ip.name
        if not out_img.exists():
            if args.copy_images:
                shutil.copy2(ip, out_img)
            else:
                try:
                    out_img.symlink_to(ip.resolve())
                except OSError:
                    shutil.copy2(ip, out_img)

        img_id = next_img_id; next_img_id += 1
        images_out.append({
            "id": img_id, "file_name": ip.name, "width": W, "height": H,
        })
        for r in records:
            annotations_out.append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": 0,
                "segmentation": [r["polygon"]],
                "area": r["area"],
                "bbox": r["bbox"],
                "score": r["score"],
                "iscrowd": 0,
            })
            next_ann_id += 1
            score_per_kept_inst.append(r["score"])

        inst_per_kept_tile.append(len(records))
        n_kept += 1

    coco = {
        "images": images_out,
        "annotations": annotations_out,
        "categories": [{"id": 0, "name": "building", "supercategory": "structure"}],
    }
    with open(args.out_coco, "w") as f:
        json.dump(coco, f)
    print(f"Wrote pseudo-COCO: {args.out_coco}")

    n_inst = len(annotations_out)
    ipt = np.array(inst_per_kept_tile) if inst_per_kept_tile else np.zeros(1)
    sc = np.array(score_per_kept_inst) if score_per_kept_inst else np.zeros(1)
    summary = [
        f"Source ckpt   : {args.checkpoint}",
        f"Image dir     : {args.image_dir}",
        f"Score thresh  : {args.score_thresh}",
        f"Area band     : [{args.min_area}, {args.max_area}]",
        f"Aspect-ratio  : [{ar_min}, {ar_max}]",
        f"Min instances per kept tile: {args.min_instances_per_tile}",
        "",
        f"Tiles processed : {len(paths)}",
        f"Tiles kept      : {n_kept}",
        f"Tiles skipped   : {n_skipped}  (no instance passed filters)",
        f"Total instances : {n_inst}",
        "",
        f"Per-kept-tile instance count: mean={ipt.mean():.2f}  median={np.median(ipt):.0f}  "
        f"min={ipt.min()}  max={ipt.max()}",
        f"Per-instance score: mean={sc.mean():.3f}  min={sc.min():.3f}  max={sc.max():.3f}",
    ]
    summary_path = args.out_coco.with_suffix("").as_posix() + "_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary))
    print("\n" + "\n".join(summary))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
