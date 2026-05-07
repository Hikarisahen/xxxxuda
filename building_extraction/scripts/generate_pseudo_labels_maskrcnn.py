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
    p.add_argument("--max-area", type=int, default=30000,
                   help="Drop instances larger than this many pixels. Real "
                        "Potsdam buildings at 5cm GSD rarely exceed ~30k px "
                        "in a 512² tile. Was 50000 — too lenient, plaza-sized "
                        "false positives slipped through.")
    p.add_argument("--max-coverage", type=float, default=0.25,
                   help="Drop instances whose mask covers more than this "
                        "fraction of the tile. Was 0.40 — too lenient.")
    p.add_argument("--min-solidity", type=float, default=0.85,
                   help="Drop instances whose mask has solidity (area/"
                        "convex_hull_area) below this. Real rectangular "
                        "rooftops are >0.9; irregular vegetation/plaza blobs "
                        "drop below ~0.8. Set 0 to disable.")
    p.add_argument("--max-vegetation-ratio", type=float, default=0.50,
                   help="IRRG-only sanity check: drop the instance if the "
                        "fraction of pixels inside the mask where R > 1.3*G "
                        "(NIR strongly above visible red, i.e. vegetation) "
                        "exceeds this. Set 1.0 to disable. Useful only on "
                        "IRRG imagery; harmless on RGB.")
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
        image_mean=model_cfg.get("image_mean"),
        image_std=model_cfg.get("image_std"),
    )
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def _vegetation_ratio_irrg(mask: np.ndarray, img: np.ndarray) -> float:
    """Fraction of mask pixels where R > 1.3*G in IRRG (vegetation signature).

    In IRRG channel mapping: R=NIR, G=visible-red, B=visible-green. Vegetation
    has very high NIR, so R is much greater than G. Bare rooftops have roughly
    balanced channels. A high ratio inside a 'building' mask means the model
    confidently outlined a vegetation patch.
    """
    sel = mask.astype(bool)
    if not sel.any():
        return 0.0
    r = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    veg_pixels = (r > 1.3 * (g + 1e-6))[sel]
    return float(veg_pixels.mean())


def _solidity(contour) -> float:
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return 0.0
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return 0.0
    return area / hull_area


def filter_instances(out: dict, img: np.ndarray,
                     score_thresh: float, mask_thresh: float,
                     min_area: int, max_area: int,
                     max_coverage: float,
                     min_solidity: float,
                     max_vegetation_ratio: float,
                     ar_min: float, ar_max: float,
                     tile_pixels: int) -> List[Dict]:
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

        # 1) Coverage check — kills the "single instance fills most of tile" failure.
        instance_pixels = int(m.sum())
        if tile_pixels > 0 and instance_pixels / tile_pixels > max_coverage:
            continue

        # 2) Vegetation-content check (IRRG-aware). Catches blobs that stray
        #    into red-ish vegetation patches even if their shape is plausible.
        if max_vegetation_ratio < 1.0:
            veg_ratio = _vegetation_ratio_irrg(m, img)
            if veg_ratio > max_vegetation_ratio:
                continue

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_area or area > max_area:
                continue

            # 3) Solidity — rectangular roofs are >0.9, ragged plaza/veg blobs drop below ~0.8.
            if min_solidity > 0:
                sol = _solidity(c)
                if sol < min_solidity:
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
                out, img,
                args.score_thresh, args.mask_binarise_thresh,
                args.min_area, args.max_area, args.max_coverage,
                args.min_solidity, args.max_vegetation_ratio,
                ar_min, ar_max, tile_pixels=H * W,
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
        f"Score thresh   : {args.score_thresh}",
        f"Area band      : [{args.min_area}, {args.max_area}]",
        f"Max coverage   : {args.max_coverage} (fraction of tile)",
        f"Min solidity   : {args.min_solidity}",
        f"Max veg ratio  : {args.max_vegetation_ratio} (IRRG R>1.3*G fraction)",
        f"Aspect-ratio   : [{ar_min}, {ar_max}]",
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
