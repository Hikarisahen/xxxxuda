#!/usr/bin/env python3
"""
Train RT-DETR for building detection on Vaihingen dataset.
Uses mmdetection framework.
"""

import os
import sys
from pathlib import Path
import argparse

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_mmdet_config():
    """Create mmdetection config for RT-DETR training."""
    config_content = '''
# RT-DETR config for building detection
model = dict(
    type='RTDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(
        type='HybridEncoder',
        in_channels=[512, 1024, 2048],
        hidden_dim=256,
        expansion=0.5,
        num_encoder_layers=1,
        num_decoder_layers=3,
        encoder_layer=dict(
            type='TransformerLayer',
            dim=256,
            num_heads=8,
            feedforward_dim=1024,
            dropout=0.0,
        ),
    ),
    bbox_head=dict(
        type='RTDETRHead',
        num_classes=1,
        embed_dims=256,
        num_decoder_layers=3,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
    ),
    train_cfg=dict(
        assigner=dict(type='HungarianAssigner', iou_calculator=dict(type='BBitmasksIoU')),
        loss=dict(
            type='RTDETRLoss',
            loss_classes_weight=2.0,
            loss_bboxes_weight=5.0,
            loss_giou_weight=2.0,
        ),
        num_classes=1,
    ),
    test_cfg=dict(
        max_per_img=100,
        score_thr=0.3,
        nms=dict(type='nms', iou_threshold=0.5),
    ),
)

dataset_type = 'CocoDataset'
data_root = 'data/vaihingen/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/val.json')
test_evaluator = val_evaluator

default_scope = 'mmdet'

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=20),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, len=1000),
    dict(type='CosineAnnealingLR', T_max=50, eta_min=1e-6, by_epoch=True),
]

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

random_seed = 42
'''
    return config_content


def create_dataset_json(image_dir, annotation_dir, output_path, split='train'):
    """Create COCO format JSON annotation file."""
    import json
    from PIL import Image

    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)

    image_paths = sorted(image_dir.glob("*.tif"))

    images = []
    annotations = []
    ann_id = 1

    categories = [{"id": 0, "name": "building", "supercategory": "structure"}]

    for img_id, img_path in enumerate(image_paths, start=1):
        img = Image.open(img_path)
        w, h = img.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })

        xml_path = annotation_dir / f"{img_path.stem}.xml"
        if xml_path.exists():
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                bbox_elem = obj.find('bndbox')
                if bbox_elem is not None:
                    xmin = int(float(bbox_elem.find('xmin').text))
                    ymin = int(float(bbox_elem.find('ymin').text))
                    xmax = int(float(bbox_elem.find('xmax').text))
                    ymax = int(float(bbox_elem.find('ymax').text))

                    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                    area = bbox[2] * bbox[3]

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 0,
                        "bbox": bbox,
                        "area": float(area),
                        "iscrowd": 0
                    })
                    ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"Created {output_path}: {len(images)} images, {len(annotations)} annotations")


def main():
    parser = argparse.ArgumentParser(description="Train RT-DETR on Vaihingen")
    parser.add_argument("--image-dir", required=True, help="Vaihingen image directory")
    parser.add_argument("--annotation-dir", required=True, help="Vaihingen annotation (XML) directory")
    parser.add_argument("--output-dir", default="checkpoints/rtdetr", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--config", default=None, help="mmdet config file (optional)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data directory for mmdet
    data_dir = output_dir / "data/vaihingen"
    data_dir.mkdir(parents=True, exist_ok=True)

    annotations_dir = data_dir / "annotations"
    annotations_dir.mkdir(exist_ok=True)

    images_link = data_dir / "images"
    if not images_link.exists():
        import shutil
        if os.path.islink(images_link):
            os.remove(images_link)
        elif images_link.exists():
            shutil.rmtree(images_link)
        os.symlink(os.path.abspath(args.image_dir), images_link)

    # Create train/val split
    image_dir = Path(args.image_dir)
    image_paths = sorted(image_dir.glob("*.tif"))
    n_train = int(len(image_paths) * 0.8)

    train_images_dir = data_dir / "train_images"
    val_images_dir = data_dir / "val_images"
    train_images_dir.mkdir(exist_ok=True)
    val_images_dir.mkdir(exist_ok=True)

    for i, img_path in enumerate(image_paths):
        import shutil
        if i < n_train:
            dst = train_images_dir / img_path.name
            if not dst.exists():
                os.symlink(os.path.abspath(img_path), dst)
        else:
            dst = val_images_dir / img_path.name
            if not dst.exists():
                os.symlink(os.path.abspath(img_path), dst)

    # Create COCO annotations
    train_ann_dir = annotations_dir / "train"
    val_ann_dir = annotations_dir / "val"
    train_ann_dir.mkdir(exist_ok=True)
    val_ann_dir.mkdir(exist_ok=True)

    print("Creating COCO format annotations...")

    train_images = list(train_images_dir.glob("*.tif"))
    val_images = list(val_images_dir.glob("*.tif"))

    # Create symlinks for annotations
    for img_path in train_images:
        ann_path = Path(args.annotation_dir) / f"{img_path.stem}.xml"
        if ann_path.exists():
            dst = train_ann_dir / f"{img_path.stem}.xml"
            if not dst.exists():
                os.symlink(os.path.abspath(ann_path), dst)

    for img_path in val_images:
        ann_path = Path(args.annotation_dir) / f"{img_path.stem}.xml"
        if ann_path.exists():
            dst = val_ann_dir / f"{img_path.stem}.xml"
            if not dst.exists():
                os.symlink(os.path.abspath(ann_path), dst)

    print("Note: You need to convert VOC annotations to COCO format.")
    print("      Or use mmdetection's VOCDataset directly.")
    print()
    print("To train with mmdetection, run:")
    print(f"  python -m mmdet.train {args.config or 'configs/rt_detr.py'} --work-dir {output_dir}")
    print()
    print("Alternative: Use YOLO which is easier to set up.")


if __name__ == "__main__":
    main()