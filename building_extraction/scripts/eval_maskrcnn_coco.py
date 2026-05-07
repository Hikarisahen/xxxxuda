#!/usr/bin/env python3
"""Evaluate a torchvision Mask R-CNN checkpoint on a COCO-format val set.

Reports both bbox and segmentation AP (full COCOeval suite: AP, AP50, AP75,
AP_small/medium/large, AR). Useful for measuring true target-domain transfer
quality — much more informative than the val_loss currently used to pick best.pt.

Important inference settings:
  * box_score_thresh forced to 0.05 (low) so pycocotools sees the full PR curve.
    Don't pass the config's 0.5 threshold here — that would mean "AP" is
    actually "AP after thresholding at 0.5", which is not the same metric.
  * NMS and max-detections kept at torchvision defaults (0.5, 100).

CLI:
    python scripts/eval_maskrcnn_coco.py \
        --config        configs/maskrcnn/maskrcnn_vaihingen.yaml \
        --checkpoint    checkpoints/maskrcnn_vaihingen/best.pt \
        --val-coco      annotations/potsdam_val_coco.json \
        --val-image-dir /home/zfx/datasets/Potsdam/IRRG_512 \
        --out-dir       eval_results/source_on_potsdam
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training.train_maskrcnn import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--val-coco", required=True, type=Path)
    p.add_argument("--val-image-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--score-thresh", type=float, default=0.05,
                   help="Low threshold so AP sees the full PR curve. Don't "
                        "raise this for evaluation.")
    p.add_argument("--mask-binarise-thresh", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(config_path: Path, checkpoint_path: Path,
               score_thresh: float, device: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]
    model = build_model(
        num_classes=mc.get("num_classes", 1),
        variant=mc.get("variant", "v2"),
        pretrained=False,
        trainable_backbone_layers=int(mc.get("trainable_backbone_layers", 3)),
        image_mean=mc.get("image_mean"),
        image_std=mc.get("image_std"),
    )
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"Partial load: missing={len(missing)} unexpected={len(unexpected)}")

    # Force low score threshold for evaluation; AP needs the full PR curve.
    model.roi_heads.score_thresh = score_thresh
    return model.to(device).eval()


def predictions_to_coco_results(pred: dict, image_id: int,
                                coco_cat_id: int,
                                mask_thresh: float) -> List[dict]:
    """Convert a single image's torchvision predictions into COCO results."""
    out: List[dict] = []
    if pred["scores"].numel() == 0:
        return out
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    boxes = pred["boxes"].cpu().numpy()        # (N, 4) xyxy
    masks_soft = pred["masks"].cpu().numpy()   # (N, 1, H, W) in [0,1]

    for i in range(len(scores)):
        if labels[i] != 1:  # 0 = background, 1 = building
            continue
        x1, y1, x2, y2 = boxes[i]
        w, h = float(x2 - x1), float(y2 - y1)
        if w <= 0 or h <= 0:
            continue

        m = (masks_soft[i, 0] >= mask_thresh).astype(np.uint8)
        # COCO RLE: needs Fortran-contiguous uint8
        rle = mask_utils.encode(np.asfortranarray(m))
        rle["counts"] = rle["counts"].decode("ascii")  # JSON-safe

        out.append({
            "image_id": image_id,
            "category_id": coco_cat_id,
            "bbox": [float(x1), float(y1), w, h],
            "score": float(scores[i]),
            "segmentation": rle,
        })
    return out


def run_eval(coco_gt: COCO, results: List[dict], iou_type: str) -> dict:
    """Run COCOeval and return the 12-metric summary dict."""
    if not results:
        print(f"  [{iou_type}] no detections — skipping COCOeval")
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0,
                "APs": 0.0, "APm": 0.0, "APl": 0.0,
                "AR1": 0.0, "AR10": 0.0, "AR100": 0.0,
                "ARs": 0.0, "ARm": 0.0, "ARl": 0.0}
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats  # standard 12-element vector
    return {
        "AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
        "APs": float(s[3]), "APm": float(s[4]), "APl": float(s[5]),
        "AR1": float(s[6]), "AR10": float(s[7]), "AR100": float(s[8]),
        "ARs": float(s[9]), "ARm": float(s[10]), "ARl": float(s[11]),
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.config, args.checkpoint, args.score_thresh, args.device)

    print(f"Loading GT: {args.val_coco}")
    coco_gt = COCO(str(args.val_coco))
    img_ids = coco_gt.getImgIds()
    print(f"  {len(img_ids)} val images, {len(coco_gt.getAnnIds())} GT instances")

    # Pull the only category id from the GT file (we use 0; pycocotools is fine
    # with it as long as both GT and DT agree).
    cat_ids = coco_gt.getCatIds()
    if len(cat_ids) != 1:
        raise RuntimeError(f"Expected 1 category, got {cat_ids}")
    coco_cat_id = cat_ids[0]

    results: List[dict] = []
    n_pred_total = 0
    with torch.no_grad():
        for img_id in tqdm(img_ids, desc="Predict"):
            meta = coco_gt.loadImgs(img_id)[0]
            img_path = args.val_image_dir / meta["file_name"]
            if not img_path.exists():
                print(f"  missing image: {img_path}")
                continue
            arr = np.array(Image.open(img_path).convert("RGB"))
            x = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0)
            x = x.to(args.device)
            pred = model([x])[0]
            recs = predictions_to_coco_results(
                pred, image_id=img_id,
                coco_cat_id=coco_cat_id,
                mask_thresh=args.mask_binarise_thresh,
            )
            results.extend(recs)
            n_pred_total += len(recs)

    print(f"\nTotal predictions: {n_pred_total}")
    pred_path = args.out_dir / "predictions.json"
    pred_path.write_text(json.dumps(results))
    print(f"Saved predictions: {pred_path}")

    print("\n========== bbox AP ==========")
    bbox_metrics = run_eval(coco_gt, results, "bbox")
    print("\n========== segm AP ==========")
    segm_metrics = run_eval(coco_gt, results, "segm")

    summary = {
        "checkpoint": str(args.checkpoint),
        "val_coco": str(args.val_coco),
        "n_val_images": len(img_ids),
        "n_gt_instances": len(coco_gt.getAnnIds()),
        "n_predictions": n_pred_total,
        "score_thresh": args.score_thresh,
        "bbox": bbox_metrics,
        "segm": segm_metrics,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n========== Summary ==========")
    print(f"  GT instances : {summary['n_gt_instances']}")
    print(f"  Predictions  : {summary['n_predictions']}")
    print(f"  bbox AP / AP50 / AP75 : "
          f"{bbox_metrics['AP']:.3f} / {bbox_metrics['AP50']:.3f} / {bbox_metrics['AP75']:.3f}")
    print(f"  segm AP / AP50 / AP75 : "
          f"{segm_metrics['AP']:.3f} / {segm_metrics['AP50']:.3f} / {segm_metrics['AP75']:.3f}")
    print(f"  Saved -> {summary_path}")


if __name__ == "__main__":
    main()
