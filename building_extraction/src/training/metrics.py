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


class BoundaryLoss(nn.Module):
    """Penalises errors on the building boundary band specifically.

    Building extraction is judged on contour quality, not just IoU — a 1-pixel
    drift along a long edge wrecks downstream vectorisation but barely moves
    Dice. We extract the boundary band of the GT mask via morphological
    gradient (dilate - erode) and apply BCE only inside that band, weighted
    by the inverse of its area so small buildings aren't drowned out by large
    ones. Used as an additive term alongside Dice + BCE.
    """

    def __init__(self, dilation: int = 3):
        super().__init__()
        self.dilation = dilation

    @staticmethod
    def _boundary_band(mask: torch.Tensor, k: int) -> torch.Tensor:
        # mask: [B, 1, H, W] in {0,1}. Returns same-shape band mask in {0,1}.
        pad = k // 2
        kernel = torch.ones(1, 1, k, k, device=mask.device, dtype=mask.dtype)
        dil = F.conv2d(mask, kernel, padding=pad).clamp_(0, 1)
        ero = 1.0 - F.conv2d(1.0 - mask, kernel, padding=pad).clamp_(0, 1)
        return (dil - ero).clamp_(0, 1)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        band = self._boundary_band(target, self.dilation)  # [B,1,H,W]
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
        # Per-image normalisation so small + large buildings contribute equally.
        denom = band.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_img = (bce * band).sum(dim=(1, 2, 3)) / denom
        return per_img.mean()


class DiceBCEBoundaryLoss(nn.Module):
    """Composite loss tuned for sharp building contours: Dice + BCE + Boundary.

    Defaults are reasonable for binary building masks; tweak weights via the
    finetune YAML if needed.
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0,
                 boundary_weight: float = 1.0, boundary_dilation: int = 3,
                 smooth: float = 1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.smooth = smooth
        self.boundary = BoundaryLoss(dilation=boundary_dilation)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred_logits, target)

        prob = torch.sigmoid(pred_logits)
        intersection = (prob * target).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        boundary = self.boundary(pred_logits, target)

        return (
            self.bce_weight * bce
            + self.dice_weight * dice_loss
            + self.boundary_weight * boundary
        )