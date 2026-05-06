#!/usr/bin/env python3
"""Render COCO ground-truth polygons over their images for visual sanity check.

Use this when you want to inspect a *labels-only* COCO file (e.g. the Potsdam
val set we just built) without spinning up a model. Produces side-by-side PNGs:
left = original image, right = image with GT polygon outlines + fill overlay.

CLI:
    python scripts/visualize_gt_coco.py \
        --coco       annotations/potsdam_val_coco.json \
        --image-dir  /home/zfx/datasets/Potsdam/IRRG_512 \
        --out-dir    /tmp/potsdam_val_viz \
        --num-samples 12
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True, type=Path)
    p.add_argument("--image-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.35,
                   help="Fill transparency for polygon overlay.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(args.coco.read_text())
    images = {img["id"]: img for img in coco["images"]}
    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    img_ids = list(images.keys())
    rng = random.Random(args.seed)
    rng.shuffle(img_ids)
    pick = img_ids[: args.num_samples]

    for img_id in tqdm(pick, desc="Rendering"):
        meta = images[img_id]
        img_path = args.image_dir / meta["file_name"]
        if not img_path.exists():
            print(f"  missing: {img_path}")
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        overlay = img.copy()
        outlines = img.copy()

        anns = anns_by_img.get(img_id, [])
        for a in anns:
            for poly in a["segmentation"]:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
                cv2.polylines(outlines, [pts], isClosed=True,
                              color=(0, 255, 0), thickness=2)

        blended = cv2.addWeighted(overlay, args.alpha, outlines,
                                  1 - args.alpha, 0)
        side_by_side = np.concatenate([img, blended], axis=1)

        # Annotate header
        header = f"{meta['file_name']}  |  {len(anns)} GT instances"
        cv2.putText(side_by_side, header, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        out_path = args.out_dir / f"{Path(meta['file_name']).stem}_gt.png"
        Image.fromarray(side_by_side).save(out_path)

    print(f"\nWrote {len(pick)} visualisations to {args.out_dir}")


if __name__ == "__main__":
    main()
