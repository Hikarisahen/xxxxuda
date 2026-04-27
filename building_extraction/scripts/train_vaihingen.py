#!/usr/bin/env python3
"""
Train on Vaihingen building dataset, then generate pseudo labels for Wuhan data.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).parent.parent))


class VaihingenDataset(Dataset):
    """Vaihingen building segmentation dataset."""

    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_paths = sorted([
            p for p in Path(image_dir).glob("*.tif")
            if not p.stem.endswith("_mask")
        ])

        print(f"Found {len(self.image_paths)} Vaihingen images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = Path(self.label_dir) / image_path.name

        image = np.array(Image.open(image_path).convert("RGB"))

        if label_path.exists():
            label = np.array(Image.open(label_path).convert("RGB"))
            # Only pure red (255,0,0) is building
            label = ((label[:, :, 0] == 255) & (label[:, :, 1] == 0) & (label[:, :, 2] == 0)).astype(np.float32)
        else:
            label = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]
        else:
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label).float()
                if label.dim() == 2:
                    label = label.unsqueeze(0)

        return image, label


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        target = target.float()

        if target.dim() == 3:
            target = target.unsqueeze(1)
        elif target.dim() == 4 and target.shape[1] == 1:
            target = target

        bce_loss = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred).view(pred.size(0), -1)
        target_view = target.view(target.size(0), -1)

        smooth = 1e-5
        intersection = (pred_sigmoid * target_view).sum(dim=1)
        union = pred_sigmoid.sum(dim=1) + target_view.sum(dim=1)
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def get_train_transforms(image_size=512):
    return A.Compose([
        A.RandomCrop(height=image_size, width=image_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=512):
    return A.Compose([
        A.CenterCrop(height=image_size, width=image_size, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0

    for images, masks in tqdm(loader, desc="Validation"):
        images = images.to(device)
        masks = masks.to(device)

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        pred = (torch.sigmoid(outputs) > 0.5).float()

        intersection = (pred * masks).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-5) / (union + 1e-5)
        total_dice += dice.mean().item()

    return total_loss / len(loader), total_dice / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train on Vaihingen dataset")
    parser.add_argument("--image-dir", type=str,
                        default="/home/zfx/datasets/Vaihingen_croped/train/Images",
                        help="Vaihingen image directory")
    parser.add_argument("--label-dir", type=str,
                        default="/home/zfx/datasets/Vaihingen_croped/train/Labels",
                        help="Vaihingen label directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Encoder name")
    parser.add_argument("--output", type=str, default="checkpoints/vaihingen",
                        help="Output checkpoint directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output, "logs"))

    train_dataset = VaihingenDataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        transform=get_train_transforms(512),
    )

    val_dataset = VaihingenDataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        transform=get_val_transforms(512),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    print(f"Model: U-Net with {args.encoder} encoder")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    criterion = DiceBCELoss(dice_weight=1.0, bce_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step()

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/dice", val_dice, epoch)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            ckpt_path = os.path.join(args.output, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice,
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    writer.close()
    print(f"Training complete! Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()