#!/usr/bin/env python3
"""Plan A: pure post-processing of U-Net semantic mask -> instance polygons.

Pipeline per image:
    prob -> Gaussian smooth -> threshold -> morphological close
         -> small-hole fill   -> connectedComponents (instance split)
         -> findContours + approxPolyDP (polygon vertex sequence)
         -> optional right-angle snap

Outputs per image:
    {stem}_mask_raw.png        binary mask straight from threshold
    {stem}_mask_clean.png      after closing + fill holes
    {stem}_instances.png       instance label map, color-coded
    {stem}_polygons.png        original image with polygons drawn on top
    {stem}_grid.png            5-panel comparison: orig | prob | raw | clean | instances+polygons
    {stem}_polygons.json       per-instance vertex list, bbox, area

Plus a top-level summary.txt with aggregate stats.

Two run modes:
    --image-dir   run inference (loads checkpoint), then post-process
    --mask-dir    skip inference, post-process existing prob/mask PNGs

Run on 5 images first to tune parameters before scaling up:
    python scripts/postprocess_plan_a.py \
        --config configs/unet_finetune.yaml \
        --checkpoint checkpoints/unet/best_unet.pt \
        --image-dir data/potsdam_pseudo_images \
        --out-dir outputs/plan_a_test \
        --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.building_model import build_building_model


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Inference                                                                    #
# --------------------------------------------------------------------------- #
def preprocess(img: np.ndarray, infer_size: int) -> torch.Tensor:
    if img.shape[0] != infer_size or img.shape[1] != infer_size:
        img = cv2.resize(img, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def load_model(config_path: str, ckpt_path: str, device: str) -> torch.nn.Module:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = build_building_model(
        arch=cfg["model"].get("arch", "smp_unet"),
        encoder_name=cfg["model"].get("encoder_name", "tu-convnext_base"),
        encoder_weights=None,
        num_classes=cfg["model"].get("num_classes", 1),
        dino_model_name=cfg["model"].get("dino_model_name", "dinov2_vitb14"),
        dino_freeze=cfg["model"].get("dino_freeze", True),
    )
    sd = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def infer_prob(model: torch.nn.Module, img: np.ndarray, infer_size: int, device: str) -> np.ndarray:
    """Returns prob map at original image resolution."""
    h, w = img.shape[:2]
    with torch.no_grad():
        x = preprocess(img, infer_size).to(device)
        prob_low = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return cv2.resize(prob_low, (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- #
# Post-processing                                                              #
# --------------------------------------------------------------------------- #
def smooth_prob(prob: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth the probability map BEFORE thresholding.

    Smoothing post-threshold loses information; smoothing pre-threshold lets
    low-confidence interior pixels (e.g. rooftop equipment) recover support
    from their high-confidence neighbours.
    """
    if sigma <= 0:
        return prob
    k = max(3, int(2 * round(3 * sigma) + 1))  # 3-sigma rule, force odd
    return cv2.GaussianBlur(prob, (k, k), sigmaX=sigma)


def morph_close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Close = dilate then erode. Connects fragments, fills small gaps."""
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    """Fill background-component holes whose area < max_hole_area.

    Trick: invert the mask, label connected background components, keep only
    the giant outer one, fill everything else.
    """
    if max_hole_area <= 0:
        return mask
    inv = (mask == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = mask.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max_hole_area:
            out[labels == i] = 1
    return out


def split_instances(mask: np.ndarray, min_area: int) -> Tuple[np.ndarray, int]:
    """connectedComponents -> filter by min_area. Returns (label_map, n_kept)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(labels)
    new_id = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            new_id += 1
            keep[labels == i] = new_id
    return keep, new_id


def extract_polygons(
    label_map: np.ndarray,
    eps_ratio: float,
    snap_right_angle: bool,
) -> List[dict]:
    """For each instance, extract simplified polygon vertices.

    eps_ratio: Douglas-Peucker tolerance, as fraction of contour perimeter.
        0.005 ~ "snap to ~10-20 vertex polygon for typical buildings".
        0.01  ~ "snap aggressively, 4-8 vertices, may lose detail".
    snap_right_angle: if True, post-snap vertex angles to nearest 90 deg axis-aligned
        relative to the polygon's dominant orientation. Cheap and helps for
        man-made buildings.
    """
    out = []
    n_instances = int(label_map.max())
    for inst_id in range(1, n_instances + 1):
        binary = (label_map == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        # Largest external contour for this instance.
        contour = max(contours, key=cv2.contourArea)
        perim = cv2.arcLength(contour, closed=True)
        eps = max(1.0, eps_ratio * perim)
        approx = cv2.approxPolyDP(contour, eps, closed=True).reshape(-1, 2)

        if snap_right_angle and len(approx) >= 4:
            approx = _orthogonal_snap(approx)

        x, y, w, h = cv2.boundingRect(contour)
        out.append({
            "id": inst_id,
            "polygon": approx.astype(int).tolist(),     # [[x,y], [x,y], ...]
            "num_vertices": int(len(approx)),
            "bbox": [int(x), int(y), int(w), int(h)],   # COCO xywh
            "area": int(binary.sum()),
            "perimeter": float(perim),
        })
    return out


def _orthogonal_snap(poly: np.ndarray) -> np.ndarray:
    """Cheap regularizer: rotate the polygon to its dominant edge angle, snap
    each vertex to the nearest axis-aligned grid (using local edge median),
    rotate back. Helps man-made buildings look less ragged.

    This is intentionally simple — for production use HiSup or a dedicated
    vector regularizer.
    """
    edges = poly[1:] - poly[:-1]
    closing_edge = (poly[0] - poly[-1]).reshape(1, 2)
    edges = np.vstack([edges, closing_edge])
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    # Fold into [0, pi/2): we want axis-aligned, so 90/180/270-deg rotations equivalent
    folded = np.mod(angles, np.pi / 2.0)
    # Use median of edge angles weighted by edge length.
    lengths = np.linalg.norm(edges, axis=1) + 1e-6
    dom_angle = np.average(folded, weights=lengths)

    c, s = np.cos(-dom_angle), np.sin(-dom_angle)
    R = np.array([[c, -s], [s, c]])
    Rinv = np.array([[c, s], [-s, c]])

    centroid = poly.mean(axis=0)
    rotated = (poly - centroid) @ R.T

    # Snap each x to its nearest neighbour x (same for y) within a tolerance,
    # so near-collinear vertical/horizontal edges become exactly axis-aligned.
    # Tolerance scales with bbox extent so it's resolution-agnostic.
    extent = max(rotated[:, 0].ptp(), rotated[:, 1].ptp(), 1.0)
    tol = max(2.0, 0.02 * extent)

    def cluster_snap(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        sorted_v = values[order]
        clusters = [[sorted_v[0]]]
        for v in sorted_v[1:]:
            if v - clusters[-1][-1] < tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        # Map each value to the mean of its cluster.
        v2c = {}
        for cl in clusters:
            m = float(np.mean(cl))
            for v in cl:
                v2c[v] = m
        out = np.array([v2c[v] for v in values])
        return out

    rotated[:, 0] = cluster_snap(rotated[:, 0])
    rotated[:, 1] = cluster_snap(rotated[:, 1])

    return rotated @ Rinv.T + centroid


# --------------------------------------------------------------------------- #
# Visualisation                                                                #
# --------------------------------------------------------------------------- #
_PALETTE = None


def _palette(n: int) -> np.ndarray:
    """Distinct random colors, deterministic across runs."""
    global _PALETTE
    if _PALETTE is None or len(_PALETTE) < n + 1:
        rng = np.random.RandomState(42)
        _PALETTE = rng.randint(64, 255, size=(max(n + 1, 64), 3), dtype=np.uint8)
        _PALETTE[0] = 0  # background = black
    return _PALETTE


def colorize_instances(label_map: np.ndarray) -> np.ndarray:
    pal = _palette(int(label_map.max()))
    return pal[label_map].astype(np.uint8)


def draw_polygons_on_image(img: np.ndarray, polygons: List[dict]) -> np.ndarray:
    out = img.copy()
    pal = _palette(len(polygons))
    for i, p in enumerate(polygons, start=1):
        pts = np.asarray(p["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        color = tuple(int(c) for c in pal[i])
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        # Vertex dots.
        for v in pts.reshape(-1, 2):
            cv2.circle(out, tuple(v), 3, color, -1)
        # Instance id label at centroid.
        cx, cy = pts.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(out, str(p["id"]), (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, str(p["id"]), (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def make_grid(
    img: np.ndarray, prob: np.ndarray, raw: np.ndarray, clean: np.ndarray,
    instance_overlay: np.ndarray,
) -> np.ndarray:
    h, w = img.shape[:2]
    prob_vis = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    prob_vis = cv2.cvtColor(prob_vis, cv2.COLOR_BGR2RGB)
    raw_vis = cv2.cvtColor((raw * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    clean_vis = cv2.cvtColor((clean * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    panels = [img, prob_vis, raw_vis, clean_vis, instance_overlay]
    panels = [cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in panels]
    return np.concatenate(panels, axis=1)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def process_one(
    img: np.ndarray,
    prob: np.ndarray,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    smoothed = smooth_prob(prob, args.smooth_sigma)
    raw = (smoothed > args.threshold).astype(np.uint8)
    closed = morph_close(raw, args.close_kernel)
    clean = fill_small_holes(closed, args.max_hole_area)
    label_map, n_inst = split_instances(clean, args.min_instance_area)
    polys = extract_polygons(label_map, args.polydp_eps_ratio, args.snap_right_angle)
    return raw, clean, label_map, polys


def iter_inputs(args):
    """Yield (stem, img, prob) tuples."""
    if args.image_dir:
        device = args.device
        model = load_model(args.config, args.checkpoint, device)
        paths = _list_images(args.image_dir)
        if args.limit and len(paths) > args.limit:
            paths = paths[: args.limit]
        for p in tqdm(paths, desc="Infer+post"):
            img = np.array(Image.open(p).convert("RGB"))
            prob = infer_prob(model, img, args.infer_size, device)
            yield p.stem, img, prob
    else:
        # mask-dir mode: pair {stem}.png images with {stem}_prob.png probs.
        # Falls back to using mask-as-prob if no _prob.png is present.
        img_paths = _list_images(args.image_dir_for_mask) if args.image_dir_for_mask else []
        img_lookup = {p.stem: p for p in img_paths}
        mask_paths = _list_images(args.mask_dir)
        if args.limit and len(mask_paths) > args.limit:
            mask_paths = mask_paths[: args.limit]
        for p in tqdm(mask_paths, desc="Post"):
            stem = p.stem.replace("_pred", "").replace("_mask", "")
            mask = np.array(Image.open(p).convert("L")).astype(np.float32) / 255.0
            img_path = img_lookup.get(stem)
            img = np.array(Image.open(img_path).convert("RGB")) if img_path \
                else np.stack([(mask * 255).astype(np.uint8)] * 3, axis=-1)
            yield stem, img, mask


def _list_images(dir_path: str) -> List[Path]:
    d = Path(dir_path)
    return sorted(
        list(d.glob("*.png")) + list(d.glob("*.jpg"))
        + list(d.glob("*.tif")) + list(d.glob("*.tiff"))
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Mode 1: full inference + post-process.
    ap.add_argument("--config", default="configs/unet_finetune.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/unet/best_unet.pt")
    ap.add_argument("--image-dir", default=None,
                    help="Run inference on these images, then post-process.")
    ap.add_argument("--infer-size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Mode 2: post-process pre-computed masks.
    ap.add_argument("--mask-dir", default=None,
                    help="Skip inference; treat each PNG here as a probability/mask map.")
    ap.add_argument("--image-dir-for-mask", default=None,
                    help="Original RGB images paired with --mask-dir, by filename stem.")

    # Output.
    ap.add_argument("--out-dir", default="outputs/plan_a")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N images (0 = all).")

    # Post-processing knobs — these are the dials worth tuning.
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold after smoothing.")
    ap.add_argument("--smooth-sigma", type=float, default=1.5,
                    help="Gaussian sigma applied to prob map BEFORE threshold. "
                         "0 disables smoothing.")
    ap.add_argument("--close-kernel", type=int, default=9,
                    help="Morphological closing kernel size (odd; <=1 disables).")
    ap.add_argument("--max-hole-area", type=int, default=1500,
                    help="Fill background components smaller than this many pixels "
                         "(0 disables).")
    ap.add_argument("--min-instance-area", type=int, default=200,
                    help="Drop building instances smaller than this.")
    ap.add_argument("--polydp-eps-ratio", type=float, default=0.005,
                    help="Douglas-Peucker tolerance, as fraction of contour perimeter. "
                         "Larger = fewer vertices.")
    ap.add_argument("--snap-right-angle", action="store_true",
                    help="Snap polygon edges to dominant orthogonal frame.")

    args = ap.parse_args()

    if not args.image_dir and not args.mask_dir:
        ap.error("Provide either --image-dir (with --checkpoint) or --mask-dir.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_images, n_polys, vertex_counts, areas = 0, 0, [], []

    for stem, img, prob in iter_inputs(args):
        raw, clean, label_map, polys = process_one(img, prob, args)
        instance_rgb = colorize_instances(label_map)
        # Blend instance colors over the original image so you can see alignment.
        instance_overlay = (0.5 * img + 0.5 * instance_rgb).clip(0, 255).astype(np.uint8)
        instance_overlay = draw_polygons_on_image(instance_overlay, polys)

        # Save artefacts.
        Image.fromarray((raw * 255).astype(np.uint8)).save(out_dir / f"{stem}_mask_raw.png")
        Image.fromarray((clean * 255).astype(np.uint8)).save(out_dir / f"{stem}_mask_clean.png")
        Image.fromarray(instance_rgb).save(out_dir / f"{stem}_instances.png")
        Image.fromarray(draw_polygons_on_image(img, polys)).save(out_dir / f"{stem}_polygons.png")
        Image.fromarray(make_grid(img, prob, raw, clean, instance_overlay)).save(
            out_dir / f"{stem}_grid.png"
        )
        with open(out_dir / f"{stem}_polygons.json", "w") as f:
            json.dump({
                "image": stem,
                "image_size": list(img.shape[:2]),
                "num_instances": len(polys),
                "instances": polys,
            }, f, indent=2)

        n_images += 1
        n_polys += len(polys)
        vertex_counts.extend([p["num_vertices"] for p in polys])
        areas.extend([p["area"] for p in polys])

    # Aggregate summary so you can compare runs at different settings.
    summary = [
        f"Images processed     : {n_images}",
        f"Total instances      : {n_polys}",
        f"  per image (mean)   : {n_polys / max(n_images, 1):.2f}",
        "",
        "Polygon vertex count (after Douglas-Peucker):",
        f"  mean / median / max: {np.mean(vertex_counts) if vertex_counts else 0:.1f} "
        f"/ {np.median(vertex_counts) if vertex_counts else 0:.0f} "
        f"/ {np.max(vertex_counts) if vertex_counts else 0}",
        "",
        "Instance pixel area:",
        f"  mean / median / min / max: "
        f"{np.mean(areas) if areas else 0:.0f} / "
        f"{np.median(areas) if areas else 0:.0f} / "
        f"{np.min(areas) if areas else 0} / "
        f"{np.max(areas) if areas else 0}",
        "",
        "Settings used:",
        f"  threshold            = {args.threshold}",
        f"  smooth_sigma         = {args.smooth_sigma}",
        f"  close_kernel         = {args.close_kernel}",
        f"  max_hole_area        = {args.max_hole_area}",
        f"  min_instance_area    = {args.min_instance_area}",
        f"  polydp_eps_ratio     = {args.polydp_eps_ratio}",
        f"  snap_right_angle     = {args.snap_right_angle}",
        "",
        "How to read the *_grid.png files (left -> right):",
        "  1. original RGB",
        "  2. raw probability map (viridis)",
        "  3. mask after smoothing + threshold (BEFORE cleanup)",
        "  4. mask after closing + hole-fill (AFTER cleanup)",
        "  5. instances colored, with polygon outlines + vertex dots",
        "",
        "Tuning hints:",
        "  * Holes still visible in panel 4 -> raise --max-hole-area or --close-kernel.",
        "  * Buildings broken into chunks  -> raise --close-kernel.",
        "  * Adjacent buildings merged     -> Plan A's known limit; switch to Plan C/D.",
        "  * Polygons too jagged           -> raise --polydp-eps-ratio (e.g. 0.01).",
        "  * Polygons miss real corners    -> lower --polydp-eps-ratio (e.g. 0.002).",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nArtefacts in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
