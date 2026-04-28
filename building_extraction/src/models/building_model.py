from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino_v2 import DinoV2Encoder
from .unet import BuildingSegmentor


class _ConvBNAct(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, k: int = 3):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )


class _DecoderBlock(nn.Module):
    """Upsample x2, concat skip, two 3x3 conv-BN-GELU."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = nn.Sequential(
            _ConvBNAct(in_c + skip_c, out_c),
            _ConvBNAct(out_c, out_c),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class DinoV2UNet(nn.Module):
    """High-resolution segmentation head over DINOv2 patch tokens.

    Why this design (vs the old broken model):
      * Old model used the [CLS] token as a 1x1 feature and upsampled to a mask,
        which discards all spatial information.
      * Here we read the *patch* tokens, reshape them to a [B,C,h,w] grid at
        stride = patch_size, and pull intermediate ViT-block features as skip
        connections so the decoder gets multi-depth context (low-level edges
        from shallow blocks, high-level semantics from deep blocks).
      * The final upsample brings the prediction back to the input resolution
        with a small refinement head that helps preserve sharp building edges.
    """

    def __init__(
        self,
        dino_model_name: str = "dinov2_vitb14",
        dino_pretrained: bool = True,
        dino_freeze: bool = True,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32),
        num_classes: int = 1,
        intermediate_layers: Optional[Tuple[int, ...]] = None,
        dino_adapted_ckpt: Optional[str] = None,
    ):
        super().__init__()

        self.encoder = DinoV2Encoder(
            model_name=dino_model_name,
            pretrained=dino_pretrained,
            freeze=dino_freeze,
            intermediate_layers=intermediate_layers,
            adapted_ckpt=dino_adapted_ckpt,
        )
        self.patch_size = self.encoder.patch_size
        feat_dim = self.encoder.feature_dim
        n_skips = len(self.encoder.intermediate_layers)

        # All ViT block outputs share the channel dim (=feat_dim) and the same
        # spatial size. We project each skip to a smaller per-stage channel
        # count so the decoder doesn't blow up in compute.
        self.skip_proj = nn.ModuleList(
            [_ConvBNAct(feat_dim, decoder_channels[0], k=1) for _ in range(n_skips)]
        )
        self.bottleneck = _ConvBNAct(feat_dim, decoder_channels[0], k=1)

        # Decoder ladder. The first 1–2 stages consume the ViT skips; later
        # stages (which are at higher spatial resolution than any ViT stage)
        # have no skip and just upsample-refine.
        self.decoder = nn.ModuleList()
        in_c = decoder_channels[0]
        for i, out_c in enumerate(decoder_channels):
            skip_c = decoder_channels[0] if i < n_skips - 1 else 0
            self.decoder.append(_DecoderBlock(in_c, skip_c, out_c))
            in_c = out_c

        # Edge refinement: a couple of 3x3 convs at full resolution help
        # produce sharper contours than a single 1x1 logit conv.
        self.refine = nn.Sequential(
            _ConvBNAct(decoder_channels[-1], decoder_channels[-1]),
            nn.Conv2d(decoder_channels[-1], num_classes, 1),
        )

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        H, W = x.shape[-2:]
        ph = (self.patch_size - H % self.patch_size) % self.patch_size
        pw = (self.patch_size - W % self.patch_size) % self.patch_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_padded, (H, W) = self._pad_to_patch_multiple(x)

        feats = self.encoder.forward_multi(x_padded)  # list of [B, C, h, w]
        # Order: shallow -> deep. Use deepest as bottleneck, shallower as skips.
        skips = [proj(f) for proj, f in zip(self.skip_proj, feats[:-1])]
        deep = self.bottleneck(feats[-1])

        # Reverse so deeper-skip is consumed first (closest to bottleneck).
        skips = list(reversed(skips))

        out = deep
        for i, block in enumerate(self.decoder):
            skip = skips[i] if i < len(skips) else None
            out = block(out, skip)

        logits = self.refine(out)

        # Crop back to the original (pre-pad) size, then resize to exactly H,W.
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits  # raw logits — apply sigmoid in loss / inference


# Backwards-compatible alias. Old code imports `BuildingExtractionModel`.
BuildingExtractionModel = DinoV2UNet


def build_building_model(
    arch: str = "smp_unet",
    encoder_name: str = "tu-convnext_base",
    encoder_weights: Optional[str] = "imagenet",
    num_classes: int = 1,
    dino_model_name: str = "dinov2_vitb14",
    dino_freeze: bool = True,
    dino_adapted_ckpt: Optional[str] = None,
    pretrained_segmentor_ckpt: Optional[str] = None,
) -> nn.Module:
    """Build a building segmentation model.

    Args:
        arch: "smp_unet" (default, strongest off-the-shelf baseline for
            high-quality contour extraction) or "dinov2_unet" (DINOv2 encoder
            + custom decoder, useful when you want to leverage DINOv2's
            transferable features directly).
        encoder_name / encoder_weights: only used by "smp_unet".
        dino_model_name / dino_freeze: only used by "dinov2_unet".
        pretrained_segmentor_ckpt: optional path to a previously trained
            checkpoint of the *same* architecture to warm-start from.
    """
    if arch == "dinov2_unet":
        model = DinoV2UNet(
            dino_model_name=dino_model_name,
            dino_pretrained=True,
            dino_freeze=dino_freeze,
            num_classes=num_classes,
            dino_adapted_ckpt=dino_adapted_ckpt,
        )
    elif arch == "smp_unet":
        model = BuildingSegmentor(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown arch: {arch!r}")

    if pretrained_segmentor_ckpt:
        sd = torch.load(pretrained_segmentor_ckpt, map_location="cpu")
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(
                f"[build_building_model] partial load: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    return model
