#!/usr/bin/env python3
"""Run a trained mmdet Mask R-CNN on a directory of images and export
per-image COCO-format predictions + raw instance masks.

The polygon/contour extraction is intentionally kept simple and matches the
shape of self_training_pipeline.py's expected `predictions` schema (one record
per instance with keys: id, image_id, image_name, polygon, area, bbox, score).

CLI:
    python scripts/predict_mmdet.py \
        --config     configs/mmdet/mask_rcnn_vaihingen.py \
        --checkpoint checkpoints/mmdet_vaihingen/best_segm_mAP.pth \
        --image-dir  data/val_images \
        --out-dir    output/mmdet_predictions \
        --score-thresh 0.5 \
        --min-area   100
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

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
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--min-area", type=int, default=100)
    p.add_argument("--max-area", type=int, default=500000)
    p.add_argument("--save-masks", action="store_true",
                   help="Also write a per-image binary union mask PNG.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def masks_to_records(pred_instances, score_thresh: float, min_area: int,
                     max_area: int) -> List[Dict]:
    """Convert mmdet pred_instances into a flat list of polygon records.

    Logic lifted (and modernised) from predict_detectron2.py:85-127. Filters
    by score, builds polygons via cv2.findContours on the binary mask, and
    drops anything outside the area band.
    """
    records: List[Dict] = []
    if len(pred_instances) == 0:
        return records

    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    bboxes = pred_instances.bboxes.cpu().numpy()
    masks = pred_instances.masks.cpu().numpy()  # bool [N, H, W]

    for i in range(len(scores)):
        if scores[i] < score_thresh:
            continue
        if labels[i] != 0:  # only "building" (single-class, but be safe)
            continue

        m = masks[i].astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_area or area > max_area:
                continue
            polygon = c.flatten().tolist()
            if len(polygon) < 6:
                continue
            x, y, w, h = cv2.boundingRect(c)
            records.append({
                "id": len(records) + 1,
                "polygon": polygon,
                "area": area,
                "bbox": [float(x), float(y), float(w), float(h)],  # COCO XYWH
                "score": float(scores[i]),
            })
    return records


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

    all_images = []
    all_anns = []
    n_inst_total = 0

    for img_id, ip in enumerate(tqdm(paths, desc="Predict"), start=1):
        img = np.array(Image.open(ip).convert("RGB"))
        h, w = img.shape[:2]

        result = inference_detector(model, img)
        pred = result.pred_instances
        records = masks_to_records(pred, args.score_thresh, args.min_area, args.max_area)

        # Per-image COCO
        all_images.append({
            "id": img_id,
            "file_name": ip.name,
            "width": w,
            "height": h,
        })
        for r in records:
            all_anns.append({
                "id": len(all_anns) + 1,
                "image_id": img_id,
                "category_id": 0,
                "segmentation": [r["polygon"]],
                "area": r["area"],
                "bbox": r["bbox"],
                "score": r["score"],
                "iscrowd": 0,
            })
        n_inst_total += len(records)

        if args.save_masks and records:
            union = np.zeros((h, w), dtype=np.uint8)
            for r in records:
                pts = np.array(r["polygon"], dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(union, [pts], 255)
            Image.fromarray(union).save(args.out_dir / f"{ip.stem}_mask.png")

    coco = {
        "images": all_images,
        "annotations": all_anns,
        "categories": [{"id": 0, "name": "building", "supercategory": "structure"}],
    }
    out_json = args.out_dir / "predictions_coco.json"
    with open(out_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\nWrote {n_inst_total} instance predictions across {len(paths)} images")
    print(f"  COCO JSON: {out_json}")
    if args.save_masks:
        print(f"  Per-image binary masks: {args.out_dir}/<stem>_mask.png")


if __name__ == "__main__":
    main()
