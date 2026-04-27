import torch
import torch.nn as nn

from .dino_v2 import DinoV2Encoder, DinoV2FeatureExtractor
from .unet import BuildingSegmentor


class BuildingExtractionModel(nn.Module):
    """Combined model for building extraction with DINOv2 encoder."""

    def __init__(
        self,
        dino_model_name: str = "dinov2_vitb14",
        dino_pretrained: bool = True,
        dino_freeze: bool = True,
        decoder_channels: tuple = (256, 128, 64, 32),
        use_pretrained_decoder: bool = False,
    ):
        super().__init__()

        self.encoder = DinoV2Encoder(
            model_name=dino_model_name,
            pretrained=dino_pretrained,
            freeze=dino_freeze,
        )

        self.decoder = nn.ModuleList()
        in_channels = self.encoder.feature_dim
        for out_ch in decoder_channels:
            self.decoder.append(nn.Sequential(
                nn.Conv2d(in_channels, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_channels = out_ch

        self.final_conv = nn.Conv2d(decoder_channels[-1], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        if len(features.shape) == 2:
            B, C = features.shape
            H = W = int(B ** 0.5) if B > 1 else int(x.shape[2] / 16)
            features = features.view(B, C, 1, 1)

        for layer in self.decoder:
            features = layer(features)
            features = torch.nn.functional.interpolate(
                features, scale_factor=2, mode="bilinear", align_corners=False
            )

        return torch.sigmoid(self.final_conv(features))


def build_building_model(
    dino_checkpoint: str = None,
    encoder_name: str = "tu-base",
    num_classes: int = 1,
) -> BuildingSegmentor:
    """Build building segmentation model."""
    model = BuildingSegmentor(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        num_classes=num_classes,
    )

    if dino_checkpoint:
        state_dict = torch.load(dino_checkpoint, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.model.encoder.load_state_dict(state_dict, strict=False)

    return model