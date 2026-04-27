import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class BuildingSegmentor(nn.Module):
    """U-Net segmentation model for building extraction."""

    def __init__(
        self,
        encoder_name: str = "tu-base",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def load_pretrained_dino_unet(dino_checkpoint_path: str = None):
        """Load U-Net with optional DINOv2 pretrained weights."""
        model = BuildingSegmentor()

        if dino_checkpoint_path is not None:
            checkpoint = torch.load(dino_checkpoint_path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                model.model.encoder.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )

        return model