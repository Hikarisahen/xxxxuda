import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class BuildingSegmentor(nn.Module):
    """U-Net segmentation model for building extraction.

    Returns raw logits — apply sigmoid in the loss / postprocessing stage.
    """

    def __init__(
        self,
        encoder_name: str = "tu-convnext_base",
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
