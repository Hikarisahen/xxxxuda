#!/bin/bash
# Download DINOv2 pretrained weights

DINO_MODELS=(
    "dinov2_vitb14"
    "dinov2_vitl14"
    "dinov2_vitg14"
)

echo "DINOv2 models will be downloaded automatically via torch.hub on first use."
echo "If you need manual download, check: https://github.com/facebookresearch/dinov2"
echo ""
echo "To verify DINOv2 installation, run:"
echo "  python -c \"import torch; model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')\""
echo ""
echo "Available models: ${DINO_MODELS[*]}"