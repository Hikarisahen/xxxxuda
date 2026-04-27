#!/usr/bin/env python3
"""
Inference script for Detectron2 Cascade Mask R-CNN building extraction.
Supports prediction on satellite images and exports instance masks.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import json

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import Instances, BoxMode


class BuildingPredictor:
    """Predictor for building instance segmentation."""

    def __init__(
        self,
        config_file: str,
        model_checkpoint: str,
        confidence_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_file)

        self.cfg.MODEL.WEIGHTS = model_checkpoint
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        self.cfg.MODEL.ROI_MASK_HEAD.SCORE_THRESH_TEST = mask_threshold
        self.cfg.TEST.DETECTED_PER_IMG = 100

        if torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = "cuda"
        else:
            self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, image: np.ndarray) -> Instances:
        """Run prediction on a single image."""
        outputs = self.predictor(image)
        return outputs["instances"]

    def predict_and_visualize(
        self,
        image: np.ndarray,
        output_path: Optional[str] = None,
        show_mask: bool = True,
        show_bbox: bool = True,
    ) -> Instances:
        """Predict and optionally save visualization."""
        instances = self.predict(image)

        if output_path:
            vis = Visualizer(
                image[:, :, ::-1],
                metadata=MetadataCatalog.get("vaihingen_val"),
                instance_mode=ColorMode.SEGMENTATION if show_mask else ColorMode.IMAGE,
            )

            vis_output = vis.draw_instance_predictions(instances.to("cpu"))
            vis_image = vis_output.get_image()[:, :, ::-1]

            cv2.imwrite(output_path, vis_image)

        return instances

    def extract_instance_masks(
        self,
        instances: Instances,
        min_area: int = 100,
        max_area: int = 500000,
    ) -> List[Dict]:
        """Extract individual building masks as polygon contours."""
        masks = []
        pred_classes = instances.pred_classes.cpu().numpy()
        pred_scores = instances.scores.cpu().numpy()
        pred_masks = instances.pred_masks.cpu().numpy()

        for i, (cls, score, mask) in enumerate(zip(pred_classes, pred_scores, pred_masks)):
            if cls != 0:
                continue

            mask_bool = mask > self.cfg.MODEL.ROI_MASK_HEAD.SCORE_THRESH_TEST
            mask_uint8 = mask_bool.astype(np.uint8)

            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue

                polygon = contour.flatten().tolist()

                bbox = list(cv2.boundingRect(contour))
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]

                masks.append({
                    "id": len(masks) + 1,
                    "polygon": polygon,
                    "area": float(area),
                    "bbox": [float(x) for x in bbox],
                    "score": float(score),
                })

        return masks


def export_to_coco_json(
    image_dir: str,
    output_path: str,
    predictions: List[Dict],
    image_id: int,
    image_info: Dict,
) -> None:
    """Export predictions to COCO annotation format."""
    annotations = []
    for pred in predictions:
        annotations.append({
            "id": pred["id"],
            "image_id": image_id,
            "category_id": 0,
            "segmentation": [pred["polygon"]],
            "area": pred["area"],
            "bbox": pred["bbox"],
            "score": pred["score"],
            "iscrowd": 0,
        })

    coco = {
        "images": [image_info],
        "annotations": annotations,
        "categories": [{"id": 0, "name": "building", "supercategory": "structure"}],
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)


def process_directory(
    predictor: BuildingPredictor,
    image_dir: str,
    output_dir: str,
    min_area: int = 100,
    max_area: int = 500000,
    export_coco: bool = False,
    export_visualization: bool = False,
) -> List[Dict]:
    """Process all images in a directory."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(image_dir.glob("*.tif")) +
        list(image_dir.glob("*.png")) +
        list(image_dir.glob("*.jpg"))
    )

    all_predictions = []
    image_id = 1

    for img_path in tqdm(image_paths, desc="Processing images"):
        image = np.array(Image.open(img_path).convert("RGB"))

        instances = predictor.predict(image)

        pred_masks = predictor.extract_instance_masks(
            instances, min_area=min_area, max_area=max_area
        )

        img_info = {
            "id": image_id,
            "file_name": img_path.name,
            "width": image.shape[1],
            "height": image.shape[0],
        }

        if export_visualization:
            vis_path = output_dir / f"{img_path.stem}_vis.png"
            predictor.predict_and_visualize(image, output_path=str(vis_path))

        if export_coco:
            coco_path = output_dir / f"{img_path.stem}_pred.json"
            export_to_coco_json(
                str(image_dir),
                str(coco_path),
                pred_masks,
                image_id,
                img_info,
            )

        for pred in pred_masks:
            pred["image_id"] = image_id
            pred["image_name"] = img_path.name
            all_predictions.append(pred)

        image_id += 1

    if all_predictions:
        summary_path = output_dir / "all_predictions.json"
        with open(summary_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"Saved {len(all_predictions)} predictions to {summary_path}")

    return all_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Detectron2 building extraction inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cascade_mask_rcnn_building.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Input image directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/predictions",
        help="Output directory"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum building area"
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=500000,
        help="Maximum building area"
    )
    parser.add_argument(
        "--export-coco",
        action="store_true",
        help="Export predictions to COCO JSON"
    )
    parser.add_argument(
        "--export-visualization",
        action="store_true",
        help="Export visualized predictions"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found: {args.model}")
        return

    predictor = BuildingPredictor(
        config_file=args.config,
        model_checkpoint=args.model,
        confidence_threshold=args.confidence_threshold,
    )

    predictions = process_directory(
        predictor=predictor,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        min_area=args.min_area,
        max_area=args.max_area,
        export_coco=args.export_coco,
        export_visualization=args.export_visualization,
    )

    print(f"\nProcessed {len(predictions)} buildings total")


if __name__ == "__main__":
    main()
