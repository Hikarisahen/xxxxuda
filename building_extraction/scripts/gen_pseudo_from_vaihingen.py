#!/usr/bin/env python3
"""
Generate pseudo labels for Wuhan data using a model trained on Vaihingen.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).parent.parent))


def morphological_cleanup(mask, kernel_size=7):
    """Connect fragmented masks and fill holes."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return (opened > 127).astype(np.float32)


def filter_by_area(mask, min_area=200, max_area=500000):
    """Filter connected components by area."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered[labels == i] = 1

    return filtered


def remove_small_holes(mask, max_hole_area=500):
    """Fill small holes inside buildings."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    inverted = cv2.bitwise_not(mask_uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    filled = mask_uint8.copy()
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < max_hole_area:
            filled[labels == i] = 255

    return (filled > 127).astype(np.float32)


class PseudoLabelGenerator:
    """Generate pseudo labels using trained model."""

    def __init__(self, checkpoint_path, encoder_name="resnet34", device="cuda"):
        self.device = device

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        print(f"Model loaded from {checkpoint_path}")
        print(f"Val Dice: {checkpoint.get('val_dice', 'N/A')}")

    def preprocess(self, image):
        """Preprocess image for model input."""
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image)

    @torch.no_grad()
    def predict(self, image, threshold=0.5):
        """Predict building mask for a single image."""
        h, w = image.shape[:2]

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        logits = self.model(image_tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

        mask = (prob > threshold).astype(np.float32)

        return mask

    def process_and_save(self, image_path, output_dir, apply_postprocessing=True):
        """Process image and save pseudo label."""
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = self.predict(image)

        if apply_postprocessing:
            mask = filter_by_area(mask, min_area=200, max_area=500000)
            mask = morphological_cleanup(mask, kernel_size=7)
            mask = remove_small_holes(mask, max_hole_area=500)
            mask = filter_by_area(mask, min_area=200, max_area=500000)

        output_path = os.path.join(output_dir, f"{Path(image_path).stem}_mask.tif")
        Image.fromarray((mask * 255).astype(np.uint8)).save(output_path)

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo labels using Vaihingen-trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--image-dir", required=True, help="Input image directory (Wuhan)")
    parser.add_argument("--output-dir", required=True, help="Output pseudo label directory")
    parser.add_argument("--encoder", default="resnet34", help="Encoder name")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    parser.add_argument("--no-postprocess", action="store_true", help="Skip postprocessing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = list(Path(args.image_dir).glob("*.tif")) + \
                  list(Path(args.image_dir).glob("*.png")) + \
                  list(Path(args.image_dir).glob("*.jpg"))

    print(f"Found {len(image_paths)} images")
    print(f"Output directory: {args.output_dir}")

    generator = PseudoLabelGenerator(
        checkpoint_path=args.checkpoint,
        encoder_name=args.encoder,
        device=device,
    )

    for image_path in tqdm(image_paths, desc="Generating pseudo labels"):
        generator.process_and_save(
            image_path,
            args.output_dir,
            apply_postprocessing=not args.no_postprocess,
        )

    print(f"Generated {len(image_paths)} pseudo labels")


if __name__ == "__main__":
    main()