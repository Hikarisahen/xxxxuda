import os
import glob
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

import segmentation_models_pytorch as smp


def morphological_cleanup(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Connect fragmented masks and fill holes using morphological operations.

    Args:
        mask: Binary mask [H, W]
        kernel_size: Size of morphological kernel

    Returns:
        Cleaned binary mask
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return (opened > 127).astype(np.float32)


def filter_by_area(mask: np.ndarray, min_area: int = 200, max_area: int = 500000) -> np.ndarray:
    """
    Filter connected components by area.

    Args:
        mask: Binary mask [H, W]
        min_area: Minimum pixel area to keep
        max_area: Maximum pixel area to keep

    Returns:
        Filtered binary mask
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered[labels == i] = 1

    return filtered


def remove_small_holes(mask: np.ndarray, max_hole_area: int = 500) -> np.ndarray:
    """
    Fill small holes inside buildings.

    Args:
        mask: Binary mask [H, W]
        max_hole_area: Maximum hole area to fill

    Returns:
        Mask with holes filled
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    inverted = cv2.bitwise_not(mask_uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    filled = mask_uint8.copy()
    h, w = mask.shape

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < max_hole_area:
            filled[labels == i] = 255

    return (filled > 127).astype(np.float32)


class SMPPseudoLabelGenerator:
    """Generate pseudo labels using Segmentation Models PyTorch."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.5,
    ):
        self.device = device
        self.threshold = threshold

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )

        self.model.to(device)
        self.model.eval()

        print(f"SMP model loaded: {encoder_name}, device: {device}")

    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate building mask for a single image.

        Args:
            image: RGB image [H, W, 3] or preprocessed tensor

        Returns:
            Binary mask [H, W]
        """
        h, w = image.shape[:2]

        if isinstance(image, np.ndarray):
            image_tensor = self._preprocess_image(image)
        else:
            image_tensor = image

        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(self.device)

            logits = self.model(image_tensor)
            prob = torch.sigmoid(logits)
            mask = (prob > self.threshold).float()

        mask = mask.squeeze().cpu().numpy()

        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.float32)

        return mask

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image)

    def process_image_file(
        self,
        image_path: str,
        output_dir: str,
        apply_postprocessing: bool = True,
    ) -> str:
        """Process a single image and save the pseudo label."""
        image = np.array(Image.open(image_path).convert("RGB"))

        mask = self.generate_mask(image)

        if apply_postprocessing:
            mask = filter_by_area(mask, min_area=200, max_area=500000)
            mask = morphological_cleanup(mask, kernel_size=7)
            mask = remove_small_holes(mask, max_hole_area=500)
            mask = filter_by_area(mask, min_area=200, max_area=500000)

        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_mask.tif")

        Image.fromarray((mask * 255).astype(np.uint8)).save(output_path)

        return output_path

    def process_batch(self, images: torch.Tensor) -> np.ndarray:
        """Process a batch of images."""
        with torch.no_grad():
            images = images.to(self.device)
            logits = self.model(images)
            probs = torch.sigmoid(logits)
            masks = (probs > self.threshold).float()

        return masks.cpu().numpy()


def generate_pseudo_labels(
    image_dir: str,
    output_dir: str,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    batch_size: int = 8,
    apply_postprocessing: bool = True,
) -> List[str]:
    """
    Generate pseudo labels for all images in a directory.

    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save pseudo labels
        encoder_name: SMP encoder name
        encoder_weights: Encoder pretrained weights
        batch_size: Number of images to process at once
        apply_postprocessing: Whether to apply morphological postprocessing

    Returns:
        List of output mask paths
    """
    os.makedirs(output_dir, exist_ok=True)

    image_paths = (
        glob.glob(os.path.join(image_dir, "*.tif")) +
        glob.glob(os.path.join(image_dir, "*.png")) +
        glob.glob(os.path.join(image_dir, "*.jpg"))
    )

    print(f"Found {len(image_paths)} images")

    generator = SMPPseudoLabelGenerator(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
    )

    output_paths = []

    for image_path in tqdm(image_paths, desc="Generating pseudo labels"):
        mask_path = generator.process_image_file(
            image_path,
            output_dir,
            apply_postprocessing=apply_postprocessing,
        )
        output_paths.append(mask_path)

    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate pseudo labels using SMP")
    parser.add_argument("--image-dir", required=True, help="Input image directory")
    parser.add_argument("--output-dir", required=True, help="Output mask directory")
    parser.add_argument("--encoder", default="resnet34", help="SMP encoder name")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--no-postprocess", action="store_true", help="Skip postprocessing")

    args = parser.parse_args()

    generate_pseudo_labels(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        encoder_name=args.encoder,
        batch_size=args.batch_size,
        apply_postprocessing=not args.no_postprocess,
    )