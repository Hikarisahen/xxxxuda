#!/usr/bin/env python3
"""
YOLO + SAM pipeline for building extraction.
YOLO detects building boxes, then SAM extracts precise masks within each box.
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

        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])

        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        other_area = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union_area = current_area + other_area - inter_area

        ious = inter_area / (union_area + 1e-6)

        indices = indices[1:][ious < iou_threshold]

    return keep


class YOLOSAMPipeline:
    """YOLO + SAM pipeline for building extraction."""

    def __init__(
        self,
        yolo_checkpoint: str,
        sam_checkpoint: str,
        sam_model_type: str = "vit_h",
        conf_thresh: float = 0.3,
        iou_thresh: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Initialize YOLO
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_checkpoint)
            self.yolo.to(device)
            self.use_yolo = True
            print(f"YOLO model loaded: {yolo_checkpoint}")
        except ImportError:
            print("Warning: ultralytics not installed. Detection will be skipped.")
            self.use_yolo = False

        # Initialize SAM
        from segment_anything import sam_model_registry, SamPredictor
        self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam_model.to(device)
        self.sam_model.eval()
        self.sam_predictor = SamPredictor(self.sam_model)
        print(f"SAM model loaded: {sam_model_type}")

    def detect_boxes(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect building boxes using YOLO.

        Returns:
            boxes: Array of [x1,y1,x2,y2]
            scores: Array of confidence scores
        """
        if not self.use_yolo:
            return np.array([]), np.array([])

        results = self.yolo.predict(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )

        boxes = []
        scores = []

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                boxes.append(xyxy)
                scores.append(conf)

        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            keep = nms(boxes, scores, self.iou_thresh)
            boxes = boxes[keep]
            scores = scores[keep]
        else:
            boxes = np.array([])
            scores = np.array([])

        return boxes, scores

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

        self.sam_predictor.set_image(image)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        points = [[center_x, center_y]]
        labels = [1]

        if box_w > 80 or box_h > 80:
            for dx, dy in [(-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3)]:
                px = int(center_x + dx * box_w / 2)
                py = int(center_y + dy * box_h / 2)
                px = np.clip(px, x1, x2 - 1)
                py = np.clip(py, y1, y2 - 1)
                points.append([px, py])
                labels.append(1)

        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        if len(masks) == 0:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)

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
            print("No buildings detected")
            return combined_mask

        for box in boxes:
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
            overlay[mask > 0] = [255, 0, 0]
            overlay_path = os.path.join(output_dir, f"{Path(image_path).stem}_overlay.png")
            Image.fromarray(overlay).save(overlay_path)

        return output_path


def main():
    parser = argparse.ArgumentParser(description="YOLO + SAM building extraction")
    parser.add_argument("--yolo-checkpoint", required=True, help="YOLO checkpoint path (.pt)")
    parser.add_argument("--sam-checkpoint", required=True, help="SAM checkpoint path")
    parser.add_argument("--image-dir", required=True, help="Input image directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--conf-thresh", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="YOLO NMS IoU threshold")
    parser.add_argument("--sam-model", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = list(Path(args.image_dir).glob("*.tif")) + \
                  list(Path(args.image_dir).glob("*.png")) + \
                  list(Path(args.image_dir).glob("*.jpg"))

    print(f"Found {len(image_paths)} images")

    pipeline = YOLOSAMPipeline(
        yolo_checkpoint=args.yolo_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        device=args.device,
    )

    for image_path in tqdm(image_paths, desc="Processing"):
        pipeline.process_image_file(str(image_path), args.output_dir, save_overlay=True)

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()