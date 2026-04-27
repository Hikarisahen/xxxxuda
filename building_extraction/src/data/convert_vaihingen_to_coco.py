#!/usr/bin/env python3
"""
Convert Vaihingen RGB masks to COCO instance segmentation format.

RGB mask format: pure red (255, 0, 0) = building pixels
Output: COCO instances JSON with polygon segmentations
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def find_contours(mask: np.ndarray) -> list:
    """Find contours from binary mask, returning list of polygons."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def contour_to_polygon(contour, epsilon_factor=0.002) -> list:
    """Convert contour to simplified polygon points."""
    if len(contour) < 3:
        return []

    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1:
        return []

    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    polygon = approx.flatten().tolist()
    return polygon


def mask_to_polygons(mask: np.ndarray, min_area: int = 100) -> list:
    """Extract individual building polygons from RGB mask."""
    building_mask = (
        (mask[:, :, 0] == 255) &
        (mask[:, :, 1] == 0) &
        (mask[:, :, 2] == 0)
    ).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        building_mask, connectivity=8
    )

    polygons = []
    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8)
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        contours = find_contours(component_mask)
        for contour in contours:
            polygon = contour_to_polygon(contour)
            if len(polygon) >= 6:
                polygons.append({
                    'polygon': polygon,
                    'area': int(area)
                })

    return polygons


def create_coco_annotations(
    image_dir: str,
    mask_dir: str,
    output_path: str,
    min_area: int = 100,
    max_area: int = 500000
) -> dict:
    """Create COCO instance segmentation annotations from Vaihingen dataset."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_paths = sorted([
        p for p in image_dir.glob("*.tif")
        if not p.stem.endswith("_mask")
    ])

    images = []
    annotations = []
    ann_id = 1

    categories = [{"id": 0, "name": "building", "supercategory": "structure"}]

    for img_id, img_path in enumerate(tqdm(image_paths, desc="Processing images")):
        img = Image.open(img_path)
        w, h = img.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })

        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = mask_dir / f"{img_path.stem}_mask.tif"

        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("RGB"))
            polygons = mask_to_polygons(mask, min_area=min_area)

            for poly_data in polygons:
                polygon = poly_data['polygon']
                area = poly_data['area']

                if area > max_area:
                    continue

                x_coords = polygon[0::2]
                y_coords = polygon[1::2]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bbox = [float(x_min), float(y_min),
                        float(x_max - x_min), float(y_max - y_min)]

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "segmentation": [polygon],
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0
                })
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"Created {output_path}: {len(images)} images, {len(annotations)} annotations")
    return coco


def create_wuhan_coco_annotations(
    image_dir: str,
    output_path: str,
    ann_file: str = None
) -> dict:
    """Create COCO annotations for Wuhan dataset (images only, no labels)."""
    image_dir = Path(image_dir)

    image_paths = sorted([
        p for p in image_dir.glob("*.png")
    ]) + sorted([
        p for p in image_dir.glob("*.tif")
    ])

    images = []
    for img_id, img_path in enumerate(tqdm(image_paths, desc="Processing images")):
        img = Image.open(img_path)
        w, h = img.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })

    categories = [{"id": 0, "name": "building", "supercategory": "structure"}]

    coco = {
        "images": images,
        "annotations": [],
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"Created {output_path}: {len(images)} images (no annotations)")
    return coco


def verify_annotations(ann_path: str) -> bool:
    """Verify COCO annotation file can be loaded correctly."""
    try:
        with open(ann_path, 'r') as f:
            coco = json.load(f)

        img_ids = set(ann['image_id'] for ann in coco['annotations'])
        valid_imgs = set(img['id'] for img in coco['images'])

        print(f"Images: {len(coco['images'])}")
        print(f"Annotations: {len(coco['annotations'])}")
        print(f"Valid image_ids in annotations: {len(img_ids & valid_imgs)}/{len(img_ids)}")
        return True
    except Exception as e:
        print(f"Error verifying annotations: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Vaihingen RGB masks to COCO format"
    )
    parser.add_argument(
        "--image-dir",
        default="/home/zfx/datasets/Vaihingen_croped/train/Images",
        help="Vaihingen image directory"
    )
    parser.add_argument(
        "--mask-dir",
        default="/home/zfx/datasets/Vaihingen_croped/train/Labels",
        help="Vaihingen mask directory"
    )
    parser.add_argument(
        "--output",
        default="annotations/vaihingen_coco.json",
        help="Output COCO JSON path"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum building area in pixels"
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=500000,
        help="Maximum building area in pixels"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify annotations after creation"
    )
    args = parser.parse_args()

    create_coco_annotations(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_path=args.output,
        min_area=args.min_area,
        max_area=args.max_area
    )

    if args.verify:
        verify_annotations(args.output)


if __name__ == "__main__":
    main()
