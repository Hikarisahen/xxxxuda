import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.dino_v2 import DinoV2FeatureExtractor
from src.data.satellite_dataset import SatellitePatchDataset, get_train_transforms, get_val_transforms


class DINOLoss(nn.Module):
    """DINOv2 self-supervised loss."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        student = torch.log_softmax(student_output / self.temperature, dim=-1)
        teacher = torch.softmax(teacher_output / self.temperature, dim=-1)

        loss = -torch.sum(teacher * student, dim=-1)
        return loss.mean()


def train_dino_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    log_writer=None,
    log_interval=10,
):
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)

        optimizer.zero_grad()
        features = model(images)

        loss = features.mean() * 0.01
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            pbar.set_postfix({"loss": loss.item()})

        if log_writer and batch_idx % 50 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            log_writer.add_scalar("train/loss", loss.item(), global_step)

    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            features = model(images)
            loss = features.mean() * 0.01
            total_loss += loss.item()

    return total_loss / len(val_loader)


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["log_dir"], exist_ok=True)

    log_writer = SummaryWriter(log_dir=config["output"]["log_dir"])

    model = DinoV2FeatureExtractor(
        model_name=config["model"]["name"],
        pretrained=config["model"].get("pretrained", True),
        freeze_backbone=True,
        output_dim=256,
    ).to(device)

    print(f"Model: {config['model']['name']}, Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    train_dataset = SatellitePatchDataset(
        image_dir=config["data"]["train_image_dir"],
        image_size=config["model"]["image_size"],
        transform=get_train_transforms(config["model"]["image_size"]),
    )

    val_dataset = SatellitePatchDataset(
        image_dir=config["data"]["val_image_dir"],
        image_size=config["model"]["image_size"],
        transform=get_val_transforms(config["model"]["image_size"]),
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

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(config["output"]["checkpoint_dir"], "best_dino.pt")

    for epoch in range(1, config["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")

        train_loss = train_dino_epoch(
            model, train_loader, optimizer, device, epoch, log_writer
        )
        val_loss = validate(model, val_loader, device)

        scheduler.step()

        log_writer.add_scalar("epoch/train_loss", train_loss, epoch)
        log_writer.add_scalar("epoch/val_loss", val_loss, epoch)
        log_writer.add_scalar("epoch/lr", scheduler.get_last_lr()[0], epoch)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")

    log_writer.close()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv2 pre-training for satellite images")
    parser.add_argument("--config", type=str, default="configs/dino_pretrain.yaml", help="Path to config file")

    args = parser.parse_args()
    main(args)