import os
import glob
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


def filter_building_masks(
    masks: List[np.ndarray],
    min_area: int = 100,
    max_area: int = 100000,
    aspect_ratio_threshold: float = 0.2,
) -> List[np.ndarray]:
    """
    Filter SAM masks to keep only building-like regions.

    Args:
        masks: List of binary masks from SAM
        min_area: Minimum pixel area to keep
        max_area: Maximum pixel area to keep
        aspect_ratio_threshold: Minimum aspect ratio (height/width) for filtering

    Returns:
        Filtered list of masks
    """
    filtered = []

    for mask in masks:
        area = mask.sum()
        if area < min_area or area > max_area:
            continue

        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            continue

        height = y_indices.max() - y_indices.min() + 1
        width = x_indices.max() - x_indices.min() + 1

        aspect_ratio = min(height, width) / max(height, width)
        if aspect_ratio < aspect_ratio_threshold:
            continue

        filtered.append(mask)

    return filtered


def combine_masks(masks: List[np.ndarray], shape: tuple) -> np.ndarray:
    """Combine multiple masks into one binary mask."""
    combined = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        combined[mask > 0] = 1
    return combined


class SAMPseudoLabelGenerator:
    """Generate pseudo labels using SAM."""

    def __init__(
        self,
        sam_model_type: str = "sam_h",
        sam_checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        points_per_side: int = 32,
    ):
        from segment_anything import sam_model_registry, SamPredictor

        self.device = device

        model_type = sam_model_type
        if sam_model_type in ["vit_h", "vit_l", "vit_b"]:
            model_type = sam_model_type
        elif sam_model_type == "sam_h":
            model_type = "vit_h"
        elif sam_model_type == "sam_l":
            model_type = "vit_l"
        elif sam_model_type == "sam_b":
            model_type = "vit_b"

        self.predictor = SamPredictor(sam_model_registry[model_type](checkpoint=sam_checkpoint))

        self.points_per_side = points_per_side

    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate building mask for a single image using SAM automatic mask generation."""
        self.predictor.set_image(image)

        h, w = image.shape[:2]

        masks, scores, _ = self.predictor.predict(
            multimask_output=False,
        )

        if len(masks) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        filtered_masks = filter_building_masks(
            [m for m in masks],
            min_area=500,
            max_area=500000,
            aspect_ratio_threshold=0.15,
        )

        return combine_masks(filtered_masks, image.shape[:2])

    def process_image_file(
        self,
        image_path: str,
        output_dir: str,
        save_overlay: bool = False,
    ) -> str:
        """Process a single image and save the pseudo label."""
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = self.generate_mask(image)

        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_mask.tif")

        Image.fromarray(mask * 255).save(output_path)

        return output_path


def generate_pseudo_labels(
    image_dir: str,
    output_dir: str,
    sam_model_type: str = "sam_h",
    sam_checkpoint: str = None,
    batch_size: int = 1,
    min_area: int = 100,
    max_area: int = 100000,
) -> List[str]:
    """
    Generate pseudo labels for all images in a directory.

    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save pseudo labels
        sam_model_type: SAM model type (sam_h, sam_l, sam_b)
        sam_checkpoint: Path to SAM checkpoint
        batch_size: Number of images to process at once
        min_area: Minimum mask area
        max_area: Maximum mask area

    Returns:
        List of output mask paths
    """
    os.makedirs(output_dir, exist_ok=True)

    image_paths = (
        glob.glob(os.path.join(image_dir, "*.tif")) +
        glob.glob(os.path.join(image_dir, "*.png")) +
        glob.glob(os.path.join(image_dir, "*.jpg"))
    )

    generator = SAMPseudoLabelGenerator(
        sam_model_type=sam_model_type,
        sam_checkpoint=sam_checkpoint,
    )

    output_paths = []

    for image_path in tqdm(image_paths, desc="Generating pseudo labels"):
        mask_path = generator.process_image_file(image_path, output_dir)
        output_paths.append(mask_path)

    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate pseudo labels using SAM")
    parser.add_argument("--image-dir", required=True, help="Input image directory")
    parser.add_argument("--output-dir", required=True, help="Output mask directory")
    parser.add_argument("--sam-model", default="sam_h", choices=["sam_h", "sam_l", "sam_b"])
    parser.add_argument("--sam-checkpoint", required=True, help="Path to SAM checkpoint")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum mask area")
    parser.add_argument("--max-area", type=int, default=100000, help="Maximum mask area")

    args = parser.parse_args()

    generate_pseudo_labels(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        sam_model_type=args.sam_model,
        sam_checkpoint=args.sam_checkpoint,
        min_area=args.min_area,
        max_area=args.max_area,
    )