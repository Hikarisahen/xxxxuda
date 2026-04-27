#!/usr/bin/env python3
"""
Building Extraction Pipeline
Coordinates: DINOv2 pretraining -> SAM pseudo-labeling -> U-Net fine-tuning -> Inference
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Building Extraction Pipeline")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["pretrain", "pseudo", "smp-pseudo", "finetune", "predict", "all"],
                        help="Pipeline stage to run")
    parser.add_argument("--config", type=str, default="configs/unet_finetune.yaml",
                        help="Config file path")
    parser.add_argument("--dino-config", type=str, default="configs/dino_pretrain.yaml",
                        help="DINOv2 config file path")
    parser.add_argument("--dino-checkpoint", type=str, default=None,
                        help="Path to DINOv2 checkpoint (skip pretrain if provided)")
    parser.add_argument("--sam-model", type=str, default="sam_h",
                        choices=["sam_h", "sam_l", "sam_b", "vit_h", "vit_l", "vit_b"],
                        help="SAM model type")
    parser.add_argument("--sam-checkpoint", type=str, default=None,
                        help="Path to SAM checkpoint")
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="SMP encoder name (for smp-pseudo stage)")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Input image directory")
    parser.add_argument("--pseudo-dir", type=str, default="data/pseudo_labels",
                        help="Output directory for pseudo labels")
    parser.add_argument("--output-dir", type=str, default="predictions",
                        help="Output directory for predictions")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint for prediction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    args = parser.parse_args()

    if args.stage == "pretrain":
        print("=" * 60)
        print("Stage 1: DINOv2 Self-Supervised Pre-training")
        print("=" * 60)
        from src.training.pretrain_dino import main as train_dino
        sys.argv = ["pretrain_dino.py", "--config", args.dino_config]
        train_dino()

    elif args.stage == "pseudo":
        print("=" * 60)
        print("Stage 2: SAM Pseudo-Label Generation")
        print("=" * 60)
        from src.data.sam_pseudolabel import generate_pseudo_labels

        if not args.sam_checkpoint:
            print("ERROR: --sam-checkpoint required for pseudo-label generation")
            return

        generate_pseudo_labels(
            image_dir=args.image_dir,
            output_dir=args.pseudo_dir,
            sam_model_type=args.sam_model,
            sam_checkpoint=args.sam_checkpoint,
        )

    elif args.stage == "smp-pseudo":
        print("=" * 60)
        print("Stage 2: SMP Pseudo-Label Generation (Recommended)")
        print("=" * 60)
        from src.data.smp_pseudolabel import generate_pseudo_labels

        generate_pseudo_labels(
            image_dir=args.image_dir,
            output_dir=args.pseudo_dir,
            encoder_name=args.encoder,
            encoder_weights="imagenet",
        )

    elif args.stage == "finetune":
        print("=" * 60)
        print("Stage 3: U-Net Fine-tuning with Pseudo Labels")
        print("=" * 60)
        from src.training.finetune_unet import main as train_unet
        sys.argv = ["finetune_unet.py", "--config", args.config,
                    "--dino-checkpoint", args.dino_checkpoint or ""]
        train_unet()

    elif args.stage == "predict":
        print("=" * 60)
        print("Stage 4: Building Extraction Inference")
        print("=" * 60)
        from src.inference.predict import main as predict
        sys.argv = ["predict.py", "--checkpoint", args.checkpoint,
                    "--image-dir", args.image_dir, "--output-dir", args.output_dir]
        predict()

    elif args.stage == "all":
        print("=" * 60)
        print("Running Full Pipeline: pretrain -> pseudo -> finetune -> predict")
        print("=" * 60)

        if not args.dino_checkpoint:
            print("\n[1/4] DINOv2 Pre-training...")
            from src.training.pretrain_dino import main as train_dino
            sys.argv = ["pretrain_dino.py", "--config", args.dino_config]
            train_dino()
            args.dino_checkpoint = "checkpoints/dino/best_dino.pt"
        else:
            print(f"[1/4] Skipping pretrain (using {args.dino_checkpoint})")

        if not args.sam_checkpoint:
            print("\n[2/4] SAM Pseudo-Label Generation...")
            from src.data.sam_pseudolabel import generate_pseudo_labels
            generate_pseudo_labels(
                image_dir=args.image_dir,
                output_dir=args.pseudo_dir,
                sam_model_type=args.sam_model,
                sam_checkpoint=args.sam_checkpoint or "sam_h.pt",
            )
        else:
            print(f"[2/4] Skipping pseudo-label (using {args.sam_checkpoint})")

        print("\n[3/4] U-Net Fine-tuning...")
        from src.training.finetune_unet import main as train_unet
        sys.argv = ["finetune_unet.py", "--config", args.config,
                    "--dino-checkpoint", args.dino_checkpoint or ""]
        train_unet()

        print("\n[4/4] Inference...")
        from src.inference.predict import main as predict
        sys.argv = ["predict.py", "--checkpoint", "checkpoints/unet/best_unet.pt",
                    "--image-dir", args.image_dir, "--output-dir", args.output_dir]
        predict()

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()