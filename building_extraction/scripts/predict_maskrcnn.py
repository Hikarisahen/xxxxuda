#!/usr/bin/env python3
"""Run a trained torchvision Mask R-CNN on a directory of images and export
per-image COCO-format predictions + (optional) raw union masks.

CLI:
    python scripts/predict_maskrcnn.py \
        --config     configs/maskrcnn/maskrcnn_vaihingen.yaml \
        --checkpoint checkpoints/maskrcnn_vaihingen/best.pt \
        --image-dir  data/val_images \
        --out-dir    output/maskrcnn_predictions \
        --score-thresh 0.5 \
        --min-area 100
"""
import argparse
import json
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
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--mask-binarise-thresh", type=float, default=0.5,
                   help="Threshold applied to soft mask probabilities "
                        "(torchvision returns float [0,1]).")
    p.add_argument("--min-area", type=int, default=100)
    p.add_argument("--max-area", type=int, default=500000)
    p.add_argument("--save-masks", action="store_true",
                   help="Also write per-image binary union mask PNGs.")
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
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[ckpt] partial load: missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval(), cfg


def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0


def output_to_records(output: dict, score_thresh: float, mask_thresh: float,
                      min_area: int, max_area: int) -> List[Dict]:
    """Filter + polygon-extract one image's torchvision Mask R-CNN output."""
    records: List[Dict] = []
    if len(output["scores"]) == 0:
        return records

    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    bboxes = output["boxes"].cpu().numpy()
    masks_soft = output["masks"].cpu().numpy()  # [N, 1, H, W] float

    for i in range(len(scores)):
        if scores[i] < score_thresh or labels[i] != 1:
            continue
        m = (masks_soft[i, 0] > mask_thresh).astype(np.uint8)
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
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(scores[i]),
            })
    return records


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.checkpoint}")
    model, cfg = load_model(args.config, args.checkpoint, args.device)

    paths = sorted(
        list(args.image_dir.glob("*.tif")) + list(args.image_dir.glob("*.tiff"))
        + list(args.image_dir.glob("*.png")) + list(args.image_dir.glob("*.jpg"))
    )
    print(f"Found {len(paths)} images under {args.image_dir}")

    images_out: List[Dict] = []
    annotations_out: List[Dict] = []
    n_inst_total = 0

    with torch.no_grad():
        for img_id, ip in enumerate(tqdm(paths, desc="Predict"), start=1):
            img = np.array(Image.open(ip).convert("RGB"))
            H, W = img.shape[:2]
            x = img_to_tensor(img).to(args.device)
            output = model([x])[0]
            records = output_to_records(
                output,
                args.score_thresh,
                args.mask_binarise_thresh,
                args.min_area,
                args.max_area,
            )

            images_out.append({
                "id": img_id, "file_name": ip.name, "width": W, "height": H,
            })
            for r in records:
                annotations_out.append({
                    "id": len(annotations_out) + 1,
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
                union = np.zeros((H, W), dtype=np.uint8)
                for r in records:
                    pts = np.array(r["polygon"], dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(union, [pts], 255)
                Image.fromarray(union).save(args.out_dir / f"{ip.stem}_mask.png")

    coco = {
        "images": images_out,
        "annotations": annotations_out,
        "categories": [{"id": 0, "name": "building", "supercategory": "structure"}],
    }
    out_json = args.out_dir / "predictions_coco.json"
    with open(out_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\nWrote {n_inst_total} instances across {len(paths)} images")
    print(f"  COCO JSON: {out_json}")
    if args.save_masks:
        print(f"  Per-image union masks: {args.out_dir}/<stem>_mask.png")


if __name__ == "__main__":
    main()
