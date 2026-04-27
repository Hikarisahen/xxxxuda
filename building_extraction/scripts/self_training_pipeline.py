#!/usr/bin/env python3
"""
Self-training pipeline for domain adaptation (Vaihingen -> Wuhan).
Uses pseudo-label based iterative training to adapt to target domain.
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def morphological_cleanup(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply morphological operations to clean up mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in the mask."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mask_filled = mask.copy()
    for contour in contours:
        cv2.drawContours(mask_filled, [contour], -1, 1, -1)

    return mask_filled


def filter_by_area(
    masks: List[Dict],
    min_area: int = 200,
    max_area: int = 500000,
) -> List[Dict]:
    """Filter masks by area constraints."""
    filtered = []
    for mask in masks:
        area = mask.get("area", 0)
        if min_area <= area <= max_area:
            filtered.append(mask)
    return filtered


def filter_by_aspect_ratio(
    masks: List[Dict],
    min_ratio: float = 0.2,
    max_ratio: float = 5.0,
) -> List[Dict]:
    """Filter masks by bounding box aspect ratio."""
    filtered = []
    for mask in masks:
        bbox = mask.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            x, y, w, h = bbox
            if w > 0 and h > 0:
                ratio = w / h
                if min_ratio <= ratio <= max_ratio:
                    filtered.append(mask)
    return filtered


def generate_pseudo_labels(
    image_dir: str,
    predictions: List[Dict],
    output_mask_dir: str,
    image_size: Optional[Tuple[int, int]] = None,
    min_confidence: float = 0.7,
) -> None:
    """Generate pseudo-label masks from predictions."""
    os.makedirs(output_mask_dir, exist_ok=True)

    predictions_by_image = defaultdict(list)
    for pred in predictions:
        predictions_by_image[pred["image_name"]].append(pred)

    for image_name, preds in tqdm(predictions_by_image.items(), desc="Generating masks"):
        img_path = Path(image_dir) / image_name
        if not img_path.exists():
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        if image_size:
            target_h, target_w = image_size
        else:
            target_h, target_w = h, w

        mask = np.zeros((h, w), dtype=np.uint8)

        for pred in preds:
            if pred.get("score", 0) < min_confidence:
                continue

            polygon = pred.get("polygon", [])
            if len(polygon) < 6:
                continue

            points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [points], 1)

        mask = morphological_cleanup(mask)
        mask = fill_holes(mask)

        mask_path = Path(output_mask_dir) / f"{Path(image_name).stem}_mask.png"
        Image.fromarray(mask * 255).save(mask_path)


def create_pseudo_label_coco(
    image_dir: str,
    output_path: str,
    predictions: List[Dict],
    min_confidence: float = 0.7,
    min_area: int = 200,
    max_area: int = 500000,
) -> None:
    """Create COCO format annotations from pseudo-labels."""
    predictions_by_image = defaultdict(list)
    for pred in predictions:
        predictions_by_image[pred["image_name"]].append(pred)

    images = []
    annotations = []
    ann_id = 1

    image_dir = Path(image_dir)

    for img_id, image_name in enumerate(
        sorted(predictions_by_image.keys()), start=1
    ):
        img_path = image_dir / image_name
        if not img_path.exists():
            continue

        img = Image.open(img_path)
        w, h = img.size

        images.append({
            "id": img_id,
            "file_name": image_name,
            "width": w,
            "height": h,
        })

        preds = predictions_by_image[image_name]
        for pred in preds:
            if pred.get("score", 0) < min_confidence:
                continue

            polygon = pred.get("polygon", [])
            if len(polygon) < 6:
                continue

            area = pred.get("area", 0)
            if area < min_area or area > max_area:
                continue

            bbox = pred.get("bbox", [0, 0, 0, 0])

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 0,
                "segmentation": [polygon],
                "area": float(area),
                "bbox": bbox,
                "score": float(pred.get("score", 0)),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "building", "supercategory": "structure"}],
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"Created {output_path}: {len(images)} images, {len(annotations)} annotations")


def merge_pseudo_labels(
    pseudo_label_paths: List[str],
    output_path: str,
) -> List[Dict]:
    """Merge multiple pseudo-label JSON files."""
    all_predictions = []

    for path in pseudo_label_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                all_predictions.extend(data.get("annotations", []))

    for i, pred in enumerate(all_predictions):
        pred["id"] = i + 1

    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    print(f"Merged {len(all_predictions)} pseudo-labels to {output_path}")
    return all_predictions


def compute_pseudo_label_statistics(predictions: List[Dict]) -> Dict:
    """Compute statistics about pseudo-labels."""
    if not predictions:
        return {}

    areas = [p.get("area", 0) for p in predictions]
    scores = [p.get("score", 0) for p in predictions]

    return {
        "total_instances": len(predictions),
        "mean_area": np.mean(areas) if areas else 0,
        "median_area": np.median(areas) if areas else 0,
        "mean_score": np.mean(scores) if scores else 0,
        "min_score": np.min(scores) if scores else 0,
        "max_score": np.max(scores) if scores else 0,
    }


class SelfTrainingPipeline:
    """Self-training pipeline for domain adaptation."""

    def __init__(
        self,
        source_image_dir: str,
        target_image_dir: str,
        config_file: str,
        base_output_dir: str = "output/self_training",
        min_confidence: float = 0.7,
        min_area: int = 200,
        max_area: int = 500000,
    ):
        self.source_image_dir = source_image_dir
        self.target_image_dir = target_image_dir
        self.config_file = config_file
        self.base_output_dir = Path(base_output_dir)
        self.min_confidence = min_confidence
        self.min_area = min_area
        self.max_area = max_area

        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def step_1_generate_pseudo_labels(
        self,
        model_checkpoint: str,
        output_dir: Optional[str] = None,
    ) -> List[Dict]:
        """Generate pseudo-labels for target domain using trained model."""
        if output_dir is None:
            output_dir = self.base_output_dir / "pseudo_labels"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        print("Step 1: Generating pseudo-labels...")
        from predict_detectron2 import BuildingPredictor

        predictor = BuildingPredictor(
            config_file=self.config_file,
            model_checkpoint=model_checkpoint,
            confidence_threshold=self.min_confidence,
        )

        predictions = []
        image_dir = Path(self.target_image_dir)

        image_paths = sorted(
            list(image_dir.glob("*.tif")) +
            list(image_dir.glob("*.png")) +
            list(image_dir.glob("*.jpg"))
        )

        for img_path in tqdm(image_paths, desc="Predicting"):
            image = np.array(Image.open(img_path).convert("RGB"))
            instances = predictor.predict(image)

            pred_masks = predictor.extract_instance_masks(
                instances,
                min_area=self.min_area,
                max_area=self.max_area,
            )

            for pred in pred_masks:
                pred["image_id"] = len(predictions) + 1
                pred["image_name"] = img_path.name
                predictions.append(pred)

        predictions = filter_by_area(predictions, self.min_area, self.max_area)
        predictions = filter_by_aspect_ratio(predictions)

        stats = compute_pseudo_label_statistics(predictions)
        print(f"Pseudo-label statistics: {stats}")

        with open(output_dir / "predictions.json", 'w') as f:
            json.dump(predictions, f, indent=2)

        return predictions

    def step_2_create_adapted_annotations(
        self,
        predictions: List[Dict],
        output_path: Optional[str] = None,
    ) -> None:
        """Create adapted COCO annotations from pseudo-labels."""
        if output_path is None:
            output_path = self.base_output_dir / "annotations" / "wuhan_adapted.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("Step 2: Creating adapted annotations...")

        predictions_by_image = defaultdict(list)
        for pred in predictions:
            predictions_by_image[pred["image_name"]].append(pred)

        images = []
        annotations = []
        ann_id = 1

        image_dir = Path(self.target_image_dir)

        for img_id, image_name in enumerate(
            sorted(predictions_by_image.keys()), start=1
        ):
            img_path = image_dir / image_name
            if not img_path.exists():
                continue

            img = Image.open(img_path)
            w, h = img.size

            images.append({
                "id": img_id,
                "file_name": image_name,
                "width": w,
                "height": h,
            })

            preds = predictions_by_image[image_name]
            for pred in preds:
                polygon = pred.get("polygon", [])
                if len(polygon) < 6:
                    continue

                area = pred.get("area", 0)
                bbox = pred.get("bbox", [0, 0, 0, 0])

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "segmentation": [polygon],
                    "area": float(area),
                    "bbox": bbox,
                    "score": float(pred.get("score", 0)),
                    "iscrowd": 0,
                })
                ann_id += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 0, "name": "building", "supercategory": "structure"}],
        }

        with open(output_path, 'w') as f:
            json.dump(coco, f, indent=2)

        print(f"Created {output_path}: {len(images)} images, {len(annotations)} annotations")

    def run(
        self,
        model_checkpoint: str,
        num_iterations: int = 3,
    ) -> None:
        """Run full self-training loop."""
        print(f"Starting self-training pipeline...")
        print(f"Source: {self.source_image_dir}")
        print(f"Target: {self.target_image_dir}")
        print(f"Iterations: {num_iterations}")

        current_checkpoint = model_checkpoint

        for iteration in range(1, num_iterations + 1):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration}/{num_iterations}")
            print(f"{'='*50}")

            iter_output_dir = self.base_output_dir / f"iteration_{iteration}"
            iter_output_dir.mkdir(parents=True, exist_ok=True)

            predictions = self.step_1_generate_pseudo_labels(
                model_checkpoint=current_checkpoint,
                output_dir=iter_output_dir / "pseudo_labels",
            )

            if not predictions:
                print("No predictions generated. Stopping.")
                break

            self.step_2_create_adapted_annotations(
                predictions=predictions,
                output_path=iter_output_dir / "annotations" / "wuhan_adapted.json",
            )

            print(f"\nTo fine-tune on adapted annotations, run:")
            print(f"python scripts/train_detectron2.py \\")
            print(f"  --train-annotation-file {iter_output_dir / 'annotations' / 'wuhan_adapted.json'} \\")
            print(f"  --output-dir output/detectron2_iter_{iteration}")

            current_checkpoint = iter_output_dir / "model_final.pth"

            if not current_checkpoint.exists():
                print("Note: No fine-tuned model yet. Run the training command above to continue.")
                break

        print("\nSelf-training pipeline complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Self-training pipeline for domain adaptation"
    )
    parser.add_argument(
        "--source-image-dir",
        type=str,
        default="/home/zfx/datasets/Vaihingen_croped/train/Images",
        help="Source domain image directory"
    )
    parser.add_argument(
        "--target-image-dir",
        type=str,
        default="/home/zfx/datasets/wuhan/train/images",
        help="Target domain image directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cascade_mask_rcnn_building.yaml",
        help="Detectron2 config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/self_training",
        help="Output directory"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence for pseudo-labels"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=200,
        help="Minimum building area"
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=500000,
        help="Maximum building area"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of self-training iterations"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate pseudo-labels without training"
    )
    args = parser.parse_args()

    if args.generate_only:
        from predict_detectron2 import BuildingPredictor

        predictor = BuildingPredictor(
            config_file=args.config,
            model_checkpoint=args.model,
            confidence_threshold=args.min_confidence,
        )

        from predict_detectron2 import process_directory

        predictions = process_directory(
            predictor=predictor,
            image_dir=args.target_image_dir,
            output_dir=args.output_dir,
            min_area=args.min_area,
            max_area=args.max_area,
        )
    else:
        pipeline = SelfTrainingPipeline(
            source_image_dir=args.source_image_dir,
            target_image_dir=args.target_image_dir,
            config_file=args.config,
            base_output_dir=args.output_dir,
            min_confidence=args.min_confidence,
            min_area=args.min_area,
            max_area=args.max_area,
        )

        pipeline.run(
            model_checkpoint=args.model,
            num_iterations=args.num_iterations,
        )


if __name__ == "__main__":
    main()
