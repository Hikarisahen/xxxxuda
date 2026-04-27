#!/usr/bin/env python3
"""
Detection + SAM pipeline for building extraction.
RT-DETR detects building boxes, then SAM extracts precise masks within each box.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import argparse

import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def box_iou(box1, box2):
    """Calculate IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)


def nms(boxes, scores, iou_threshold=0.5):
    """Non-maximum suppression."""
    if len(boxes) == 0:
        return []

    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]

        ious = np.array([box_iou(current_box, b) for b in other_boxes])

        indices = indices[1:][ious < iou_threshold]

    return keep


def filter_boxes(boxes, scores, min_area=100, max_area=500000, aspect_ratio_min=0.2, aspect_ratio_max=5.0):
    """Filter boxes by area and aspect ratio."""
    filtered_boxes = []
    filtered_scores = []

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)

        if area < min_area or area > max_area:
            continue

        w = x2 - x1
        h = y2 - y1
        aspect_ratio = min(w, h) / max(w, h)

        if aspect_ratio < aspect_ratio_min or aspect_ratio > aspect_ratio_max:
            continue

        filtered_boxes.append(box)
        filtered_scores.append(score)

    return np.array(filtered_boxes), np.array(filtered_scores)


class DetectionSAMPipeline:
    """Detection + SAM pipeline for building extraction."""

    def __init__(
        self,
        detector_checkpoint: str,
        sam_checkpoint: str,
        detector_config: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        detector_score_thresh: float = 0.3,
        sam_model_type: str = "vit_h",
    ):
        self.device = device
        self.score_thresh = detector_score_thresh

        # Initialize SAM
        from segment_anything import sam_model_registry, SamPredictor
        self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam_model.to(device)
        self.sam_model.eval()
        self.sam_predictor = SamPredictor(self.sam_model)

        # Initialize detector
        self.detector = None
        self.detector_checkpoint = detector_checkpoint

        if detector_config is None:
            # Try to use RT-DETR
            try:
                from mmdet.apis import init_detector, inference_detector
                from mmdet.registry import VISUALIZERS
                from mmdet.structures import DetDataSample

                self.detector = init_detector(
                    detector_config or "configs/rt_detr.py",
                    detector_checkpoint,
                    device=device
                )
                self.inference_detector = inference_detector
                self.use_mmdet = True
                print("Using mmdetection detector")
            except ImportError:
                print("Warning: mmdetection not available. Using fallback detection.")
                self.use_mmdet = False
        else:
            try:
                from mmdet.apis import init_detector, inference_detector
                self.detector = init_detector(detector_config, detector_checkpoint, device=device)
                self.inference_detector = inference_detector
                self.use_mmdet = True
            except ImportError:
                self.use_mmdet = False

    def detect_boxes(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect building boxes in image.

        Returns:
            boxes: Array of [x1,y1,x2,y2]
            scores: Array of confidence scores
        """
        if self.use_mmdet and self.detector is not None:
            result = self.inference_detector(self.detector, image)

            if hasattr(result, 'pred_instances'):
                boxes = result.pred_instances.bboxes.cpu().numpy()
                scores = result.pred_instances.scores.cpu().numpy()
            else:
                boxes = np.array([])
                scores = np.array([])

            if len(boxes) > 0:
                boxes, scores = filter_boxes(boxes, scores)
                keep = nms(boxes, scores, iou_threshold=0.5)
                boxes = boxes[keep]
                scores = scores[keep]

            return boxes, scores

        else:
            # Fallback: Use simple threshold-based detection
            # This is a placeholder - in practice you would use the trained detector
            print("Warning: Using fallback detection. Please train a detector model first.")
            return np.array([]), np.array([])

    def extract_mask_with_sam(
        self,
        image: np.ndarray,
        box: np.ndarray,
    ) -> np.ndarray:
        """
        Extract mask within a bounding box using SAM.

        Args:
            image: RGB image [H, W, 3]
            box: Bounding box [x1, y1, x2, y2]

        Returns:
            Binary mask [H, W]
        """
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        box_h = y2 - y1
        box_w = x2 - x1

        if box_h <= 0 or box_w <= 0:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Set image for SAM
        self.sam_predictor.set_image(image)

        # Use center point of box as prompt
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Try multiple points for better coverage
        points = []
        labels = []

        # Center point
        points.append([center_x, center_y])
        labels.append(1)

        # Add corner/edge points for large boxes
        if box_w > 100 or box_h > 100:
            offsets = [
                (-0.3, -0.3), (0.3, -0.3),
                (-0.3, 0.3), (0.3, 0.3)
            ]
            for dx, dy in offsets:
                px = int(center_x + dx * box_w / 2)
                py = int(center_y + dy * box_h / 2)
                px = np.clip(px, x1, x2 - 1)
                py = np.clip(py, y1, y2 - 1)
                points.append([px, py])
                labels.append(1)

        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        # Get mask from SAM
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        if len(masks) == 0:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Select best mask based on score
        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        # Crop mask to box region and resize
        mask_cropped = mask[y1:y2, x1:x2]

        # Create full-size mask
        full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = (mask_cropped > 0).astype(np.uint8)

        return full_mask

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process single image through detection + SAM pipeline.

        Args:
            image: RGB image [H, W, 3]

        Returns:
            Combined building mask [H, W]
        """
        boxes, scores = self.detect_boxes(image)

        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if len(boxes) == 0:
            return combined_mask

        for box in tqdm(boxes, desc="SAM masks"):
            mask = self.extract_mask_with_sam(image, box)
            combined_mask = np.maximum(combined_mask, mask)

        return combined_mask

    def process_image_file(
        self,
        image_path: str,
        output_dir: str,
        save_overlay: bool = False,
    ) -> str:
        """Process single image and save result."""
        image = np.array(Image.open(image_path).convert("RGB"))

        mask = self.process_image(image)

        output_path = os.path.join(output_dir, f"{Path(image_path).stem}_mask.tif")
        Image.fromarray(mask * 255).save(output_path)

        if save_overlay:
            overlay = image.copy()
            overlay[mask > 0] = [255, 0, 0]  # Red overlay
            overlay_path = os.path.join(output_dir, f"{Path(image_path).stem}_overlay.png")
            Image.fromarray(overlay).save(overlay_path)

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Detection + SAM building extraction")
    parser.add_argument("--detector-config", help="Detector config file (mmdet)")
    parser.add_argument("--detector-checkpoint", required=True, help="Detector checkpoint path")
    parser.add_argument("--sam-checkpoint", required=True, help="SAM checkpoint path")
    parser.add_argument("--image-dir", required=True, help="Input image directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--score-thresh", type=float, default=0.3, help="Detection score threshold")
    parser.add_argument("--sam-model", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = list(Path(args.image_dir).glob("*.tif")) + \
                  list(Path(args.image_dir).glob("*.png")) + \
                  list(Path(args.image_dir).glob("*.jpg"))

    print(f"Found {len(image_paths)} images")

    pipeline = DetectionSAMPipeline(
        detector_checkpoint=args.detector_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        detector_config=args.detector_config,
        device=args.device,
        detector_score_thresh=args.score_thresh,
        sam_model_type=args.sam_model,
    )

    for image_path in tqdm(image_paths, desc="Processing"):
        pipeline.process_image_file(str(image_path), args.output_dir, save_overlay=True)

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()