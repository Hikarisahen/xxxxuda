import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_PATCH_SIZE_BY_NAME = {
    "dinov2_vits14": 14,
    "dinov2_vitb14": 14,
    "dinov2_vitl14": 14,
    "dinov2_vitg14": 14,
}


class DinoV2Encoder(nn.Module):
    """DINOv2 encoder that returns spatial patch features [B, C, h, w].

    Optionally returns features from multiple intermediate transformer blocks,
    which is what a U-Net style decoder needs for skip connections.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        pretrained: bool = True,
        freeze: bool = True,
        intermediate_layers: Optional[Tuple[int, ...]] = None,
        adapted_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.patch_size = _PATCH_SIZE_BY_NAME.get(model_name, 14)

        # facebookresearch/dinov2 hub entry returns the full ViT with
        # forward_features() and get_intermediate_layers().
        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=pretrained
        )
        self.feature_dim = self.model.embed_dim

        if adapted_ckpt:
            sd = torch.load(adapted_ckpt, map_location="cpu")
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(
                    f"[DinoV2Encoder] adapted ckpt partial load: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )

        depth = len(self.model.blocks)
        if intermediate_layers is None:
            # Pick 4 evenly spaced blocks for a U-Net style decoder.
            self.intermediate_layers = tuple(
                int(round(depth * r)) - 1 for r in (0.25, 0.5, 0.75, 1.0)
            )
        else:
            self.intermediate_layers = tuple(intermediate_layers)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def _grid_hw(self, x: torch.Tensor) -> Tuple[int, int]:
        H, W = x.shape[-2:]
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input H,W={H},{W} must be divisible by patch_size={self.patch_size}"
            )
        return H // self.patch_size, W // self.patch_size

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return last-block patch tokens reshaped to [B, C, h, w]."""
        h, w = self._grid_hw(x)
        out = self.model.forward_features(x)
        tokens = out["x_norm_patchtokens"]  # [B, h*w, C]
        B, N, C = tokens.shape
        return tokens.transpose(1, 2).reshape(B, C, h, w).contiguous()

    @torch.amp.autocast("cuda", enabled=False)
    def forward_multi(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return spatial features from each selected intermediate block.

        Each tensor is [B, C, h, w] at the same spatial resolution h,w
        (ViT keeps token count constant across blocks).
        """
        h, w = self._grid_hw(x)
        feats = self.model.get_intermediate_layers(
            x, n=self.intermediate_layers, reshape=False, norm=True
        )
        out = []
        for f in feats:
            B, N, C = f.shape
            out.append(f.transpose(1, 2).reshape(B, C, h, w).contiguous())
        return out

    def get_output_dim(self) -> int:
        return self.feature_dim


class DinoV2FeatureExtractor(nn.Module):
    """DINOv2 + projection head, used for self-supervised pretraining only.

    Returns globally-pooled features through a projection MLP. Do NOT use this
    as a segmentation backbone — use DinoV2Encoder directly.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        pretrained: bool = True,
        freeze_backbone: bool = False,
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
        feat = self.encoder(x)              # [B, C, h, w]
        feat = feat.mean(dim=(2, 3))        # GAP -> [B, C]
        return self.projection(feat)

    @property
    def output_dim(self) -> int:
        return self.projection[-1].out_features
