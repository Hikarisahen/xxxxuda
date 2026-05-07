"""Train torchvision Mask R-CNN for building instance segmentation.

Single entry point handling both:
  * source-domain training (Vaihingen GT)
  * self-training (Vaihingen GT + Potsdam pseudo-labels via data.extra_train)

The training loop follows the torchvision detection reference implementation
style: SGD + linear warmup + multi-step LR drops + grad clip.

Usage:
    # Source training
    python -m src.training.train_maskrcnn \
        --config configs/maskrcnn/maskrcnn_vaihingen.yaml

    # Self-training round 1 (warm-start from source ckpt)
    python -m src.training.train_maskrcnn \
        --config configs/maskrcnn/maskrcnn_st_round1.yaml \
        --resume-from checkpoints/maskrcnn_vaihingen/best.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.coco_instance_dataset import CocoBuildingDataset, collate_fn
from src.data.maskrcnn_transforms import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(num_classes: int = 1, variant: str = "v2",
                pretrained: bool = True,
                trainable_backbone_layers: int = 3,
                image_mean: Optional[list] = None,
                image_std: Optional[list] = None) -> torch.nn.Module:
    """Build a torchvision Mask R-CNN with single-class heads.

    The torchvision API counts background as class 0, so we always pass
    num_classes + 1 to the model and let label==1 mean 'building'.

    image_mean/image_std: if provided, override the GeneralizedRCNNTransform's
        normalisation constants. Use this to inject IRRG-specific stats
        computed by scripts/compute_channel_stats.py — the ImageNet RGB
        defaults are a poor match for IRRG (R=NIR) input.
    """
    if variant == "v2":
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights,
        )
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    elif variant == "v1":
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights,
        )
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    else:
        raise ValueError(f"Unknown model variant: {variant!r}")

    # Replace the box and mask heads for our 1-class problem.
    n_total = num_classes + 1  # +1 for background
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, n_total)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, hidden, n_total)

    # Override normalisation constants if provided. torchvision's default is
    # ImageNet RGB ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]); for IRRG
    # input that's a poor match. Replace both transform.image_mean and
    # transform.image_std atomically so they stay consistent.
    if image_mean is not None:
        model.transform.image_mean = list(image_mean)
        print(f"  Overriding image_mean: {model.transform.image_mean}")
    if image_std is not None:
        model.transform.image_std = list(image_std)
        print(f"  Overriding image_std:  {model.transform.image_std}")
    return model


# ---------------------------------------------------------------------------
# Dataset assembly (handles source + optional extra mixing)
# ---------------------------------------------------------------------------
def build_train_dataset(cfg: dict):
    train_tf = get_train_transforms()
    primary = CocoBuildingDataset(
        ann_file=cfg["data"]["train_ann_file"],
        image_dir=cfg["data"]["train_image_dir"],
        transforms=train_tf,
    )
    extras = cfg["data"].get("extra_train") or []
    if not extras:
        print(f"Train dataset: {len(primary)} samples (source only)")
        return primary

    parts = [primary]
    for entry in extras:
        ds = CocoBuildingDataset(
            ann_file=entry["ann_file"],
            image_dir=entry["image_dir"],
            transforms=train_tf,
        )
        parts.append(ds)
        print(f"  + extra train source: {entry['ann_file']}  ({len(ds)} samples)")
    concat = ConcatDataset(parts)
    print(f"Train dataset: {len(concat)} samples total ({len(parts)} sources)")
    return concat


def build_val_dataset(cfg: dict):
    return CocoBuildingDataset(
        ann_file=cfg["data"]["val_ann_file"],
        image_dir=cfg["data"]["val_image_dir"],
        transforms=get_val_transforms(),
    )


# ---------------------------------------------------------------------------
# LR schedule (linear warmup → multi-step)
# ---------------------------------------------------------------------------
def make_lr_lambda(warmup_iters: int, warmup_factor: float,
                   milestones_iter: list[int], gamma: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            alpha = step / max(1, warmup_iters)
            return warmup_factor * (1 - alpha) + alpha
        # Apply gamma per milestone passed
        factor = 1.0
        for m in milestones_iter:
            if step >= m:
                factor *= gamma
        return factor
    return lr_lambda


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, epoch,
                    grad_clip: float, writer, global_step: int) -> tuple[float, int]:
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc=f"Train ep {epoch}")
    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {global_step}: {loss_dict}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        total += loss.item()
        global_step += 1

        # Per-step logging (every ~25 steps to keep tqdm + tb readable)
        if global_step % 25 == 0:
            writer.add_scalar("train/loss_total", loss.item(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    return total / max(1, len(loader)), global_step


# ---------------------------------------------------------------------------
# Validation (uses train mode briefly to compute loss; see notes inline)
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device) -> dict:
    """Compute val loss + simple per-image stats.

    torchvision Mask R-CNN only returns loss in train() mode. To get a clean
    val loss we put the model in train() mode but disable grad. Predictions
    for visualisation are produced separately by predict_maskrcnn.py.
    """
    model.train()  # required for loss computation; we still use no_grad
    total = 0.0
    n_inst_pred = 0
    n_inst_gt = 0
    for images, targets in tqdm(loader, desc="Val"):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += sum(loss_dict.values()).item()
        for t in targets:
            n_inst_gt += int(t["labels"].numel())

    # Eval-mode predictions for instance counts
    model.eval()
    for images, _ in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
        for o in outputs:
            n_inst_pred += int((o["scores"] >= 0.5).sum().item())

    return {
        "val_loss": total / max(1, len(loader)),
        "pred_inst_total": n_inst_pred,
        "gt_inst_total": n_inst_gt,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Optional checkpoint to warm-start the model from "
                         "(e.g. source ckpt for self-training rounds).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  torch={torch.__version__} torchvision={torchvision.__version__}")

    # ---- Output dirs ----
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    log_dir = Path(cfg["output"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir.as_posix())

    # ---- Model ----
    model_cfg = cfg["model"]
    model = build_model(
        num_classes=model_cfg.get("num_classes", 1),
        variant=model_cfg.get("variant", "v2"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        trainable_backbone_layers=int(model_cfg.get("trainable_backbone_layers", 3)),
        image_mean=model_cfg.get("image_mean"),
        image_std=model_cfg.get("image_std"),
    )
    if args.resume_from:
        print(f"Warm-starting from {args.resume_from}")
        sd = torch.load(args.resume_from, map_location="cpu")
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  partial load: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)

    # ---- Data ----
    train_ds = build_train_dataset(cfg)
    val_ds = build_val_dataset(cfg)
    print(f"Val dataset:   {len(val_ds)} samples")

    train_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=int(train_cfg["num_workers"]) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(train_cfg["batch_size"]) // 2),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ---- Optimizer + LR schedule ----
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=float(train_cfg["lr"]),
        momentum=float(train_cfg["momentum"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    iters_per_epoch = max(1, len(train_loader))
    epochs = int(train_cfg["epochs"])
    milestones_iter = [int(m) * iters_per_epoch for m in train_cfg.get("lr_milestones", [])]
    lr_lambda = make_lr_lambda(
        warmup_iters=int(train_cfg.get("warmup_iters", 0)),
        warmup_factor=float(train_cfg.get("warmup_factor", 0.001)),
        milestones_iter=milestones_iter,
        gamma=float(train_cfg.get("lr_gamma", 0.1)),
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Train loop ----
    best_val = math.inf
    global_step = 0
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            grad_clip, writer, global_step,
        )
        metrics = validate(model, val_loader, device)
        dt = time.time() - t0

        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", metrics["val_loss"], epoch)
        writer.add_scalar("epoch/pred_inst_total", metrics["pred_inst_total"], epoch)
        writer.add_scalar("epoch/gt_inst_total", metrics["gt_inst_total"], epoch)

        print(
            f"\nEpoch {epoch}/{epochs}  ({dt:.1f}s)\n"
            f"  Train loss : {train_loss:.4f}\n"
            f"  Val loss   : {metrics['val_loss']:.4f}\n"
            f"  Pred inst  : {metrics['pred_inst_total']}  vs GT {metrics['gt_inst_total']}"
        )

        # Save checkpoint every epoch + best-by-val-loss
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  Saved new best (val_loss={best_val:.4f}) -> {ckpt_dir/'best.pt'}")

    writer.close()
    print(f"\nDone. Best val_loss={best_val:.4f}.  Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
