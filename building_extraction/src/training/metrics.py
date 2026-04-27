import torch
import torch.nn as nn
import torch.nn.functional as F


def DiceScore(pred, target, smooth=1e-5):
    """
    Calculate Dice coefficient.

    Args:
        pred: Predicted binary mask [B, 1, H, W]
        target: Ground truth binary mask [B, 1, H, W]
        smooth: Smoothing factor

    Returns:
        Dice coefficient
    """
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()


def IoU(pred, target, smooth=1e-5):
    """
    Calculate Intersection over Union (IoU).

    Args:
        pred: Predicted binary mask [B, 1, H, W]
        target: Ground truth binary mask [B, 1, H, W]
        smooth: Smoothing factor

    Returns:
        IoU score
    """
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def F1Score(pred, target, smooth=1e-5):
    """
    Calculate F1 score.

    Args:
        pred: Predicted binary mask [B, 1, H, W]
        target: Ground truth binary mask [B, 1, H, W]
        smooth: Smoothing factor

    Returns:
        F1 score
    """
    intersection = (pred * target).sum(dim=(2, 3))
    precision = (intersection + smooth) / (pred.sum(dim=(2, 3)) + smooth)
    recall = (intersection + smooth) / (target.sum(dim=(2, 3)) + smooth)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1.mean()


def PixelAccuracy(pred, target):
    """Calculate pixel accuracy."""
    correct = (pred == target).float()
    return correct.mean()


class SegmentationMetrics:
    """Track and compute segmentation metrics over multiple batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_dice = 0
        self.total_iou = 0
        self.total_f1 = 0
        self.total_accuracy = 0
        self.num_batches = 0

    def update(self, pred, target):
        pred = pred.detach()
        target = target.detach()

        self.total_dice += DiceScore(pred, target).item()
        self.total_iou += IoU(pred, target).item()
        self.total_f1 += F1Score(pred, target).item()
        self.total_accuracy += PixelAccuracy(pred, target).item()
        self.num_batches += 1

    def get_metrics(self):
        if self.num_batches == 0:
            return {}
        return {
            "dice": self.total_dice / self.num_batches,
            "iou": self.total_iou / self.num_batches,
            "f1": self.total_f1 / self.num_batches,
            "accuracy": self.total_accuracy / self.num_batches,
        }


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for segmentation."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_prob = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight

        return (focal_weight * bce).mean()