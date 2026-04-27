import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.building_model import BuildingSegmentor, build_building_model
from src.data.satellite_dataset import SatellitePatchDataset
from src.data.augmentations import get_building_augmentations
from src.training.metrics import DiceScore, IoU, F1Score


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = DiceBCELoss(
            dice_weight=config["loss"]["dice_weight"],
            bce_weight=config["loss"]["bce_weight"],
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )

        total_steps = len(train_loader) * config["training"]["epochs"]
        warmup_steps = int(total_steps * config["training"].get("warmup_ratio", 0.1))

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=len(train_loader), T_mult=2
        )

        self.best_dice = 0
        os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
        self.log_dir = config["output"]["log_dir"]
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_dice = 0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            with torch.no_grad():
                pred_mask = (torch.sigmoid(outputs) > 0.5).float()
                dice = DiceScore(pred_mask, masks)
                total_dice += dice.item()

            pbar.set_postfix({
                "loss": loss.item(),
                "dice": dice.item(),
            })

            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("train/loss", loss.item(), global_step)
            self.writer.add_scalar("train/dice", dice.item(), global_step)
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], global_step)

        return total_loss / len(self.train_loader), total_dice / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        total_f1 = 0

        for images, masks in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            total_loss += loss.item()

            pred_mask = (torch.sigmoid(outputs) > 0.5).float()
            dice = DiceScore(pred_mask, masks)
            iou = IoU(pred_mask, masks)
            f1 = F1Score(pred_mask, masks)

            total_dice += dice.item()
            total_iou += iou.item()
            total_f1 += f1.item()

        avg_loss = total_loss / len(self.val_loader)
        avg_dice = total_dice / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)
        avg_f1 = total_f1 / len(self.val_loader)

        self.writer.add_scalar("val/loss", avg_loss, epoch)
        self.writer.add_scalar("val/dice", avg_dice, epoch)
        self.writer.add_scalar("val/iou", avg_iou, epoch)
        self.writer.add_scalar("val/f1", avg_f1, epoch)

        return avg_loss, avg_dice, avg_iou, avg_f1

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_dice": self.best_dice,
        }

        ckpt_path = os.path.join(
            self.config["output"]["checkpoint_dir"],
            f"checkpoint_epoch_{epoch}.pt",
        )
        torch.save(checkpoint, ckpt_path)

        if is_best:
            best_path = os.path.join(
                self.config["output"]["checkpoint_dir"],
                "best_unet.pt",
            )
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def train(self):
        for epoch in range(1, self.config["training"]["epochs"] + 1):
            train_loss, train_dice = self.train_epoch(epoch)
            val_loss, val_dice, val_iou, val_f1 = self.validate(epoch)

            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Val F1: {val_f1:.4f}")

            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.save_checkpoint(epoch, is_best=True)

        self.writer.close()
        print("Training complete!")


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_building_model(
        dino_checkpoint=args.dino_checkpoint,
        encoder_name=config["model"]["encoder_name"],
        num_classes=config["model"]["num_classes"],
    )

    print(f"Model loaded, parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    train_transform = get_building_augmentations("train", config["data"]["image_size"])
    val_transform = get_building_augmentations("val", config["data"]["image_size"])

    train_dataset = SatellitePatchDataset(
        image_dir=config["data"]["train_images"],
        mask_dir=config["data"].get("train_masks"),
        image_size=config["data"]["image_size"],
        transform=train_transform,
    )

    val_dataset = SatellitePatchDataset(
        image_dir=config["data"]["val_images"],
        mask_dir=config["data"].get("val_masks"),
        image_size=config["data"]["image_size"],
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    trainer = Trainer(model, train_loader, val_loader, config, device)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune U-Net for building extraction")
    parser.add_argument("--config", type=str, default="configs/unet_finetune.yaml", help="Path to config file")
    parser.add_argument("--dino-checkpoint", type=str, default=None, help="Path to DINOv2 checkpoint")

    args = parser.parse_args()
    main(args)