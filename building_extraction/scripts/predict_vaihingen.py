#!/usr/bin/env python3
"""
Predict on Vaihingen validation set.
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

import segmentation_models_pytorch as smp

sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_metrics(pred, target):
    """Calculate Dice, IoU, Precision, Recall."""
    pred_binary = (pred > 0.5).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)

    intersection = (pred_binary * target_binary).sum()
    pred_sum = pred_binary.sum()
    target_sum = target_binary.sum()
    union = pred_sum + target_sum - intersection

    dice = (2 * intersection + 1e-5) / (union + 1e-5)
    iou = (intersection + 1e-5) / (union + 1e-5)
    precision = (intersection + 1e-5) / (pred_sum + 1e-5)
    recall = (intersection + 1e-5) / (target_sum + 1e-5)

    return dice, iou, precision, recall


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Predict on Vaihingen validation")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model")
    parser.add_argument("--image-dir", default="/home/zfx/datasets/Vaihingen_croped/val/Images",
                        help="Vaihingen validation image directory")
    parser.add_argument("--label-dir", default="/home/zfx/datasets/Vaihingen_croped/val/Labels",
                        help="Vaihingen validation label directory")
    parser.add_argument("--output-dir", default="val_predictions",
                        help="Output directory for predictions")
    parser.add_argument("--encoder", default="resnet34", help="Encoder name")
    parser.add_argument("--save-visual", action="store_true", help="Save visual comparison")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model from {args.checkpoint}")
    if "val_dice" in checkpoint:
        print(f"Training Val Dice: {checkpoint['val_dice']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = sorted(Path(args.image_dir).glob("*.tif"))

    total_dice, total_iou, total_precision, total_recall = 0, 0, 0, 0
    num_images = len(image_paths)

    print(f"\nPredicting on {num_images} images...")

    for image_path in tqdm(image_paths):
        label_path = Path(args.label_dir) / image_path.name

        # Read image
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]

        # Preprocess
        image_float = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_float = (image_float - mean) / std
        image_tensor = torch.from_numpy(image_float.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        # Predict
        with torch.no_grad():
            logits = model(image_tensor)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Resize to original size
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

        # Calculate metrics if label exists
        if label_path.exists():
            label = np.array(Image.open(label_path).convert("RGB"))
            # Only pure red (255,0,0) is building
            label = ((label[:, :, 0] == 255) & (label[:, :, 1] == 0) & (label[:, :, 2] == 0)).astype(np.float32)

            dice, iou, precision, recall = calculate_metrics(prob, label)
            total_dice += dice
            total_iou += iou
            total_precision += precision
            total_recall += recall

        # Save prediction
        pred_binary = (prob > 0.5).astype(np.uint8) * 255
        Image.fromarray(pred_binary).save(
            os.path.join(args.output_dir, f"{image_path.stem}_pred.tif")
        )

        # Save visual comparison
        if args.save_visual and label_path.exists():
            label_binary = (label * 255).astype(np.uint8)
            comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
            comparison[:, :w] = image
            comparison[:, w:2*w] = np.stack([label_binary, label_binary, label_binary], axis=2)
            comparison[:, 2*w:] = np.stack([pred_binary, pred_binary, pred_binary], axis=2)
            Image.fromarray(comparison).save(
                os.path.join(args.output_dir, f"{image_path.stem}_compare.png")
            )

    # Print average metrics
    if num_images > 0:
        print(f"\n{'='*50}")
        print(f"Average Metrics on {num_images} images:")
        print(f"  Dice:     {total_dice/num_images:.4f}")
        print(f"  IoU:      {total_iou/num_images:.4f}")
        print(f"  Precision:{total_precision/num_images:.4f}")
        print(f"  Recall:   {total_recall/num_images:.4f}")
        print(f"{'='*50}")

        # Save metrics to file
        with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
            f.write(f"Dice: {total_dice/num_images:.4f}\n")
            f.write(f"IoU: {total_iou/num_images:.4f}\n")
            f.write(f"Precision: {total_precision/num_images:.4f}\n")
            f.write(f"Recall: {total_recall/num_images:.4f}\n")


if __name__ == "__main__":
    main()