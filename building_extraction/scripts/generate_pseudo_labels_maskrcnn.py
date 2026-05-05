#!/usr/bin/env python3
"""Generate instance-level pseudo-labels on a target-domain image directory
using a trained torchvision Mask R-CNN. Output is a COCO JSON ready to be
fed back as a training dataset by train_maskrcnn.py via data.extra_train.

Per-tile pipeline:
    1. forward -> per-instance score, bbox, soft mask
    2. drop instances with score < --score-thresh (default 0.85)
    3. binarise mask, polygonise via cv2.findContours
    4. per-instance area + aspect-ratio filter
    5. drop tiles with fewer than --min-instances-per-tile kept instances
    6. emit a single COCO JSON

CLI:
    python scripts/generate_pseudo_labels_maskrcnn.py \
        --config       configs/maskrcnn/maskrcnn_vaihingen.yaml \
        --checkpoint   checkpoints/maskrcnn_vaihingen/best.pt \
        --image-dir    /home/zfx/datasets/Potsdam/IRRG_512 \
        --out-image-dir building_extraction/data/potsdam_pseudo_images \
        --out-coco     building_extraction/data/potsdam_pseudo_coco.json \
        --score-thresh 0.85 --min-area 300
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_maskrcnn import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image-dir", required=True, type=Path)
    p.add_argument("--out-image-dir", required=True, type=Path)
    p.add_argument("--out-coco", required=True, type=Path)
    p.add_argument("--score-thresh", type=float, default=0.85)
    p.add_argument("--mask-binarise-thresh", type=float, default=0.5)
    p.add_argument("--min-area", type=int, default=300)
    p.add_argument("--max-area", type=int, default=300000)
    p.add_argument("--aspect-ratio-range", type=float, nargs=2, default=(0.2, 5.0),
                   metavar=("MIN", "MAX"))
    p.add_argument("--min-instances-per-tile", type=int, default=1)
    p.add_argument("--copy-images", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(config_path: str, checkpoint_path: str, device: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    model = build_model(
        num_classes=model_cfg.get("num_classes", 1),
        variant=model_cfg.get("variant", "v2"),
        pretrained=False,
        trainable_backbone_layers=int(model_cfg.get("trainable_backbone_layers", 3)),
    )
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def filter_instances(out: dict, score_thresh: float, mask_thresh: float,
                     min_area: int, max_area: int,
                     ar_min: float, ar_max: float) -> List[Dict]:
    records: List[Dict] = []
    if len(out["scores"]) == 0:
        return records
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    bboxes = out["boxes"].cpu().numpy()
    masks_soft = out["masks"].cpu().numpy()

    for i in range(len(scores)):
        if scores[i] < score_thresh or labels[i] != 1:
            continue
        m = (masks_soft[i, 0] > mask_thresh).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.config, args.checkpoint, args.device)

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

    with torch.no_grad():
        for ip in tqdm(paths, desc="Pseudo-label"):
            img = np.array(Image.open(ip).convert("RGB"))
            H, W = img.shape[:2]
            x = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().div_(255.0)
            x = x.to(args.device)
            out = model([x])[0]
            records = filter_instances(
                out, args.score_thresh, args.mask_binarise_thresh,
                args.min_area, args.max_area, ar_min, ar_max,
            )
            if len(records) < args.min_instances_per_tile:
                n_skipped += 1
                continue

            # Stage the image into the kept-tiles directory.
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
    print(f"\nWrote pseudo-COCO: {args.out_coco}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
