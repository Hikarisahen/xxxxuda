"""Domain-adaptive DINO-style self-supervised pretraining.

Why this exists at all:
  Official DINOv2 weights are pretrained on natural images. For best
  transfer to overhead/satellite imagery we do a short, *real* DINO-style
  self-distillation pass on unlabeled target-domain tiles. This shifts the
  encoder's distribution toward overhead views without needing labels.

What this implements:
  * Student-teacher self-distillation (Caron et al., DINO).
  * Teacher = EMA of student.
  * Two global crops per image → cross-view prediction loss.
  * Centering buffer + softmax sharpening to prevent collapse.

What this is NOT:
  * Not a from-scratch pretrain. We start from official DINOv2 weights
    and only adapt them. 25 epochs on a domain-target image set is
    plenty for the adaptation; DO NOT try to train DINOv2 from random
    init with this script.
"""

import os
import sys
import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import glob

sys.path.append(str(Path(__file__).parent.parent))

from src.models.dino_v2 import DinoV2Encoder


class TwoCropPatchDataset(Dataset):
    """Yields two independent augmented views of each image."""

    def __init__(self, image_dir: str, image_size: int = 224):
        self.paths = (
            sorted(glob.glob(os.path.join(image_dir, "*.tif")))
            + sorted(glob.glob(os.path.join(image_dir, "*.png")))
            + sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        )
        self.tf = A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.4, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        v1 = self.tf(image=img)["image"]
        v2 = self.tf(image=img)["image"]
        return v1, v2


class DINOHead(nn.Module):
    """3-layer MLP + L2-norm bottleneck + weight-normalized linear classifier.

    This matches the original DINO head and is what makes the self-distillation
    objective meaningful (vs naive MSE on raw features).
    """

    def __init__(self, in_dim: int, out_dim: int = 65536, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DinoBackboneWithHead(nn.Module):
    def __init__(self, model_name: str, head_out_dim: int = 65536):
        super().__init__()
        self.backbone = DinoV2Encoder(
            model_name=model_name, pretrained=True, freeze=False
        )
        self.head = DINOHead(self.backbone.feature_dim, out_dim=head_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)              # [B, C, h, w]
        feat = feat.mean(dim=(2, 3))         # GAP -> [B, C]
        return self.head(feat)


class DINOLoss(nn.Module):
    """Cross-entropy between teacher (sharpened, centered) and student logits."""

    def __init__(self, out_dim: int, teacher_temp: float = 0.04,
                 student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
        # student_out / teacher_out: each is [2B, D] — two views stacked along batch dim.
        s = student_out / self.student_temp
        t = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1).detach()

        # Cross-view targets: view1 of teacher supervises view2 of student, and vice versa.
        B = s.shape[0] // 2
        s1, s2 = s[:B], s[B:]
        t1, t2 = t[:B], t[B:]

        loss = 0.5 * (
            torch.sum(-t1 * F.log_softmax(s2, dim=-1), dim=-1).mean()
            + torch.sum(-t2 * F.log_softmax(s1, dim=-1), dim=-1).mean()
        )

        # Update center (EMA over teacher outputs).
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        return loss


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, m: float) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)


def train_one_epoch(student, teacher, loss_fn, loader, opt, device, epoch,
                    teacher_momentum: float, writer=None) -> float:
    student.train(); teacher.eval()
    total = 0.0
    pbar = tqdm(loader, desc=f"DINO pretrain epoch {epoch}")
    for step, (v1, v2) in enumerate(pbar):
        v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
        x = torch.cat([v1, v2], dim=0)

        with torch.no_grad():
            t_out = teacher(x)
        s_out = student(x)

        loss = loss_fn(s_out, t_out)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
        opt.step()

        ema_update(student, teacher, teacher_momentum)

        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if writer and step % 50 == 0:
            writer.add_scalar("train/loss", loss.item(), epoch * len(loader) + step)

    return total / max(1, len(loader))


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["log_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["output"]["log_dir"])

    student = DinoBackboneWithHead(cfg["model"]["name"]).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    out_dim = student.head.last_layer.weight.shape[0]
    loss_fn = DINOLoss(out_dim=out_dim).to(device)

    ds = TwoCropPatchDataset(
        image_dir=cfg["data"]["train_image_dir"],
        image_size=cfg["model"]["image_size"],
    )
    loader = DataLoader(
        ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=cfg["training"]["num_workers"], pin_memory=True, drop_last=True,
    )
    print(f"Pretrain samples: {len(ds)}")

    opt = optim.AdamW(student.parameters(),
                      lr=cfg["training"]["lr"],
                      weight_decay=cfg["training"]["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["training"]["epochs"])

    base_m = 0.996  # teacher EMA momentum; ramped to 1.0 over training
    epochs = cfg["training"]["epochs"]
    ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], "dino_adapted.pt")

    for epoch in range(1, epochs + 1):
        # Linear ramp of teacher momentum 0.996 -> 1.0
        m = 1.0 - (1.0 - base_m) * (1.0 - (epoch - 1) / max(1, epochs - 1))
        loss = train_one_epoch(student, teacher, loss_fn, loader, opt, device,
                               epoch, teacher_momentum=m, writer=writer)
        sched.step()

        writer.add_scalar("epoch/loss", loss, epoch)
        writer.add_scalar("epoch/lr", sched.get_last_lr()[0], epoch)
        writer.add_scalar("epoch/teacher_momentum", m, epoch)
        print(f"Epoch {epoch}/{epochs}  loss={loss:.4f}  lr={sched.get_last_lr()[0]:.2e}  m={m:.4f}")

        # Save the *teacher backbone* — that's the more stable encoder for downstream use.
        torch.save({
            "epoch": epoch,
            "model_state_dict": teacher.backbone.model.state_dict(),
            "loss": loss,
        }, ckpt_path)

    writer.close()
    print(f"Done. Adapted DINOv2 backbone saved to {ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Domain-adaptive DINOv2 pretraining")
    p.add_argument("--config", type=str, default="configs/dino_pretrain.yaml")
    main(p.parse_args())
