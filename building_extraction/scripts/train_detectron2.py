#!/usr/bin/env python3
"""
Train Detectron2 Cascade Mask R-CNN for building instance segmentation.
Supports training on Vaihingen dataset and pseudo-label based domain adaptation.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from detectron2.engine import DefaultTrainer, launch
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.model_zoo import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer


def setup_config(
    config_file: str,
    train_dataset_name: str,
    val_dataset_name: str,
    output_dir: str,
    lr: float = 0.0001,
    max_iter: int = 90000,
    batch_size: int = 8,
    num_workers: int = 4,
    eval_period: int = 5000,
    checkpoint_period: int = 5000,
) -> get_cfg():
    """Setup Detectron2 configuration."""
    cfg = get_cfg()

    # First load base model zoo config
    base_config = get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.merge_from_file(base_config)

    # Then merge custom overrides
    if config_file:
        cfg.merge_from_file(config_file)

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)

    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.SOLVER.WARMUP_ITERS = 1000

    cfg.TEST.EVAL_PERIOD = eval_period

    cfg.DATALOADER.NUM_WORKERS = num_workers

    cfg.SEED = 42

    return cfg


def register_datasets(
    train_image_dir: str,
    train_annotation_file: str,
    val_image_dir: str,
    val_annotation_file: str,
    dataset_name: str = "vaihingen",
    annotation_format: str = "coco",
) -> tuple:
    """Register datasets in Detectron2 DatasetCatalog."""
    train_dataset_name = f"{dataset_name}_train"
    val_dataset_name = f"{dataset_name}_val"

    register_coco_instances(
        train_dataset_name,
        {},
        train_annotation_file,
        train_image_dir
    )

    register_coco_instances(
        val_dataset_name,
        {},
        val_annotation_file,
        val_image_dir
    )

    return train_dataset_name, val_dataset_name


class Trainer(DefaultTrainer):
    """Custom Trainer for building instance segmentation."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(
            dataset_name,
            cfg,
            distributed=False,
            output_dir=output_folder,
            max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE,
            use_fast_impl=True,
        )

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_hooks(self):
        hooks = super().build_hooks()
        return hooks


def main():
    parser = argparse.ArgumentParser(
        description="Train Detectron2 Cascade Mask R-CNN for building extraction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cascade_mask_rcnn_building.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--train-image-dir",
        type=str,
        default="/home/zfx/datasets/Vaihingen_croped/train/Images",
        help="Training image directory"
    )
    parser.add_argument(
        "--train-annotation-file",
        type=str,
        default="annotations/vaihingen_train_coco.json",
        help="Training COCO annotation file"
    )
    parser.add_argument(
        "--val-image-dir",
        type=str,
        default="/home/zfx/datasets/Vaihingen_croped/train/Images",
        help="Validation image directory"
    )
    parser.add_argument(
        "--val-annotation-file",
        type=str,
        default="annotations/vaihingen_val_coco.json",
        help="Validation COCO annotation file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/detectron2",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=90000,
        help="Maximum training iterations"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Images per batch"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers"
    )
    parser.add_argument(
        "--eval-period",
        type=int,
        default=5000,
        help="Evaluation period"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pretrained weights (COCO or other)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs"
    )
    args = parser.parse_args()

    setup_logger(name="detectron2")
    logger = setup_logger()

    logger.info(f"Arguments: {args}")

    if not os.path.exists(args.train_annotation_file):
        logger.error(f"Training annotation file not found: {args.train_annotation_file}")
        logger.info("Please run convert_vaihingen_to_coco.py first to create annotations")
        return

    train_dataset_name, val_dataset_name = register_datasets(
        train_image_dir=args.train_image_dir,
        train_annotation_file=args.train_annotation_file,
        val_image_dir=args.val_image_dir,
        val_annotation_file=args.val_annotation_file,
    )

    cfg = setup_config(
        config_file=args.config,
        train_dataset_name=train_dataset_name,
        val_dataset_name=val_dataset_name,
        output_dir=args.output_dir,
        lr=args.lr,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_period=args.eval_period,
    )

    if args.weights:
        logger.info(f"Loading pretrained weights from {args.weights}")
        cfg.MODEL.WEIGHTS = args.weights
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )

    if args.eval_only:
        logger.info("Running evaluation only...")
        model = Trainer.build_model(cfg)
        checkpointer = DetectionCheckpointer(
            model, save_dir=cfg.OUTPUT_DIR
        )
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if verify_results(cfg, res):
            logger.info("Evaluation passed!")
        return

    trainer = Trainer(cfg)

    if args.resume:
        logger.info("Resuming from checkpoint...")
        trainer.resume_or_load(resume=args.resume)
    else:
        trainer.resume_or_load(resume=False)

    logger.info("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
