import torch
import torch.nn as nn


class DinoV2Encoder(nn.Module):
    """DINOv2 encoder for extracting visual features from satellite images."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze

        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.feature_dim = self.model.embed_dim

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            features: [B, hidden_dim] or [B, H*W, hidden_dim] depending on model
        """
        return self.model(x)

    def get_output_dim(self) -> int:
        return self.feature_dim


class DinoV2FeatureExtractor(nn.Module):
    """DINOv2 with projection head for feature extraction."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        output_dim: int = 256,
    ):
        super().__init__()
        self.encoder = DinoV2Encoder(
            model_name=model_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )

        self.projection = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 512),
            nn.GELU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if len(features.shape) == 3:
            features = features.mean(dim=1)
        return self.projection(features)

    @property
    def output_dim(self) -> int:
        return self.projection[-1].out_features