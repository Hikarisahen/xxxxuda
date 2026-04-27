import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.building_model import build_building_model
from src.data.augmentations import get_light_augmentations


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    model = build_building_model()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(
    model,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """Predict building mask for a single image."""
    h, w = image.shape[:2]

    transform = get_light_augmentations(image_size=max(h, w))
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    output = model(image_tensor)

    mask = torch.sigmoid(output).squeeze().cpu().numpy()

    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    binary_mask = (mask_resized > threshold).astype(np.uint8)

    return binary_mask


def predict_directory(
    model,
    image_dir: str,
    output_dir: str,
    device: torch.device,
    threshold: float = 0.5,
    mask_suffix: str = "_building",
    extensions: tuple = (".tif", ".png", ".jpg"),
) -> List[str]:
    """Predict building masks for all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(image_dir).glob(f"*{ext}"))
        image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))

    output_paths = []

    for image_path in tqdm(image_paths, desc="Predicting"):
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = predict_image(model, image, device, threshold)

        output_name = image_path.stem + f"{mask_suffix}.png"
        output_path = os.path.join(output_dir, output_name)

        Image.fromarray(mask * 255).save(output_path)
        output_paths.append(output_path)

    return output_paths


def predict_tiled(
    model,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 128,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Predict building mask using tiled inference for large images.

    Args:
        model: Trained model
        image: Input image [H, W, 3]
        device: Device to run inference
        tile_size: Size of each tile
        overlap: Overlap between tiles
        threshold: Binarization threshold

    Returns:
        Binary mask [H, W]
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap

    mask_sum = np.zeros((h, w), dtype=np.float32)
    mask_count = np.zeros((h, w), dtype=np.float32)

    transform = get_light_augmentations(image_size=tile_size)

    y_positions = list(range(0, h - tile_size + 1, stride))
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)

    x_positions = list(range(0, w - tile_size + 1, stride))
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)

    for y in y_positions:
        for x in x_positions:
            tile = image[y:y+tile_size, x:x+tile_size]

            augmented = transform(image=tile)
            tile_tensor = augmented["image"].unsqueeze(0).to(device)

            output = model(tile_tensor)
            tile_mask = torch.sigmoid(output).squeeze().cpu().numpy()

            mask_sum[y:y+tile_size, x:x+tile_size] += tile_mask
            mask_count[y:y+tile_size, x:x+tile_size] += 1

    mask_count[mask_count == 0] = 1
    mask_avg = mask_sum / mask_count

    return (mask_avg > threshold).astype(np.uint8)


def extract_polygons(mask: np.ndarray, simplify_tolerance: float = 1.0) -> List:
    """
    Extract building polygons from binary mask.

    Args:
        mask: Binary mask [H, W]
        simplify_tolerance: Douglas-Peucker simplification tolerance

    Returns:
        List of polygons (each polygon is a list of (x, y) points)
    """
    mask_uint8 = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if len(contour) < 4:
            continue

        contour = contour.squeeze()
        if len(contour.shape) == 1:
            continue

        polygon = [(int(pt[0]), int(pt[1])) for pt in contour]

        epsilon = simplify_tolerance * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        simplified = [(int(pt[0]), int(pt[1])) for pt in approx.reshape(-1, 2)]

        polygons.append(simplified)

    return polygons


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")

    if args.image:
        print(f"Predicting single image: {args.image}")
        image = np.array(Image.open(args.image).convert("RGB"))

        if args.tiled:
            mask = predict_tiled(model, image, device, tile_size=args.tile_size, threshold=args.threshold)
        else:
            mask = predict_image(model, image, device, threshold=args.threshold)

        output_path = args.output or args.image.replace(Path(args.image).suffix, "_building.png")
        Image.fromarray(mask * 255).save(output_path)
        print(f"Saved mask to {output_path}")

        if args.save_polygons:
            polygons = extract_polygons(mask, simplify_tolerance=args.simplify_tolerance)
            import json
            with open(output_path.replace(".png", "_polygons.json"), "w") as f:
                json.dump(polygons, f)
            print(f"Saved {len(polygons)} polygons")

    elif args.image_dir:
        output_paths = predict_directory(
            model,
            args.image_dir,
            args.output_dir or "predictions",
            device,
            threshold=args.threshold,
            mask_suffix=args.mask_suffix,
        )
        print(f"Predicted {len(output_paths)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building extraction inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image-dir", type=str, help="Directory with images")
    parser.add_argument("--output", type=str, help="Output path for single image")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for large images")
    parser.add_argument("--tiled", action="store_true", help="Use tiled inference")
    parser.add_argument("--save-polygons", action="store_true", help="Extract polygon outlines")
    parser.add_argument("--simplify-tolerance", type=float, default=1.0, help="Polygon simplification tolerance")
    parser.add_argument("--mask-suffix", type=str, default="_building", help="Suffix for output masks")

    args = parser.parse_args()
    main(args)