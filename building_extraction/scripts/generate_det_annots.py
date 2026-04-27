#!/usr/bin/env python3
"""
Generate detection annotations (bounding boxes) from Vaihingen building masks.
Outputs VOC format XML annotations.
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2


def mask_to_boxes(mask, min_area=100, max_area=500000, aspect_ratio_min=0.2, aspect_ratio_max=5.0):
    """
    Convert binary mask to list of bounding boxes.

    Args:
        mask: Binary mask [H, W]
        min_area: Minimum box area in pixels
        max_area: Maximum box area in pixels
        aspect_ratio_min: Minimum width/height ratio
        aspect_ratio_max: Maximum width/height ratio

    Returns:
        List of (x_min, y_min, x_max, y_max) tuples
    """
    mask_uint8 = (mask * 255).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    boxes = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < min_area or area > max_area:
            continue

        aspect_ratio = min(w, h) / max(w, h)
        if aspect_ratio < aspect_ratio_min:
            continue
        if aspect_ratio > aspect_ratio_max:
            continue

        boxes.append((x, y, x + w, y + h))

    return boxes


def create_voc_xml(image_path, image_shape, boxes, class_name="building"):
    """
    Create VOC format XML annotation.

    Args:
        image_path: Path to the image file
        image_shape: (height, width, channels)
        boxes: List of (x_min, y_min, x_max, y_max)
        class_name: Class name for the objects

    Returns:
        XML string
    """
    height, width, channels = image_shape
    filename = Path(image_path).name

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<annotation>',
        f'  <folder>Annotations</folder>',
        f'  <filename>{filename}</filename>',
        '  <source>',
        '    <database>Vaihingen</database>',
        '  </source>',
        '  <size>',
        f'    <width>{width}</width>',
        f'    <height>{height}</height>',
        f'    <depth>{channels}</depth>',
        '  </size>',
        '  <segmented>0</segmented>',
    ]

    for box in boxes:
        x_min, y_min, x_max, y_max = box
        xml_lines.extend([
            '  <object>',
            f'    <name>{class_name}</name>',
            '    <pose>Unspecified</pose>',
            '    <truncated>0</truncated>',
            '    <difficult>0</difficult>',
            '    <bndbox>',
            f'      <xmin>{x_min}</xmin>',
            f'      <ymin>{y_min}</ymin>',
            f'      <xmax>{x_max}</xmax>',
            f'      <ymax>{y_max}</ymax>',
            '    </bndbox>',
            '  </object>',
        ])

    xml_lines.append('</annotation>')

    return '\n'.join(xml_lines)


def create_coco_annotations(image_paths, masks_dict, class_id=0):
    """
    Create COCO format annotations.

    Args:
        image_paths: List of image paths
        masks_dict: Dict mapping image_name to binary mask
        class_id: Class ID for buildings

    Returns:
        Dict with COCO format annotations
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": class_id,
            "name": "building",
            "supercategory": "structure"
        }]
    }

    annotation_id = 1

    for img_id, image_path in enumerate(image_paths, start=1):
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]

        filename = Path(image_path).name
        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "height": h,
            "width": w
        })

        if filename in masks_dict:
            mask = masks_dict[filename]
            boxes = mask_to_boxes(mask)

            for box in boxes:
                x_min, y_min, x_max, y_max = box
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = (x_max - x_min) * (y_max - y_min)

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": float(area),
                    "iscrowd": 0
                })
                annotation_id += 1

    return coco


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate detection annotations from building masks")
    parser.add_argument("--label-dir", required=True, help="Directory containing label masks (.tif)")
    parser.add_argument("--output-dir", required=True, help="Output directory for annotations")
    parser.add_argument("--format", default="voc", choices=["voc", "coco"], help="Annotation format")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum box area")
    parser.add_argument("--max-area", type=int, default=500000, help="Maximum box area")
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(label_dir.glob("*.tif"))
    print(f"Found {len(label_paths)} label files")

    all_boxes = []
    masks_dict = {}

    for label_path in tqdm(label_paths, desc="Processing labels"):
        label = np.array(Image.open(label_path).convert("RGB"))

        # Only pure red (255,0,0) is building
        mask = (label[:, :, 0] == 255) & (label[:, :, 1] == 0) & (label[:, :, 2] == 0)
        mask = mask.astype(np.float32)

        boxes = mask_to_boxes(
            mask,
            min_area=args.min_area,
            max_area=args.max_area
        )
        all_boxes.append((label_path.name, boxes))

        # Store mask for COCO format
        masks_dict[label_path.name] = mask

        if args.format == "voc":
            xml_content = create_voc_xml(
                str(label_path),
                label.shape,
                boxes
            )

            xml_path = output_dir / f"{label_path.stem}.xml"
            with open(xml_path, "w") as f:
                f.write(xml_content)

    print(f"\nGenerated {len(all_boxes)} annotations in VOC format")
    print(f"Output directory: {output_dir}")

    total_boxes = sum(len(boxes) for _, boxes in all_boxes)
    print(f"Total boxes: {total_boxes}")

    if args.format == "coco":
        import json
        image_paths = [label_dir / name for name, _ in all_boxes]
        coco = create_coco_annotations(image_paths, masks_dict)

        json_path = output_dir / "annotations.json"
        with open(json_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"Saved COCO annotations to {json_path}")

    # Print statistics
    if all_boxes:
        num_buildings = [len(boxes) for _, boxes in all_boxes]
        print(f"\nBuildings per image: min={min(num_buildings)}, max={max(num_buildings)}, avg={np.mean(num_buildings):.1f}")


if __name__ == "__main__":
    main()