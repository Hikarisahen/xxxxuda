# Mask R-CNN (R-50 FPN) for Vaihingen building instance segmentation.
# Inherits the standard 1x COCO Mask R-CNN setup and overrides:
#   * num_classes = 1 (just "building")
#   * dataset paths -> our pre-cropped 512x512 tiles + COCO annotations
#   * LR scaled down for single-GPU batch
#
# Run with mmdet 3.x:
#   python -m mmdet.tools.train configs/mmdet/mask_rcnn_vaihingen.py \
#       --work-dir checkpoints/mmdet_vaihingen
# (or `mim train mmdet ...` if you use openmim)

_base_ = [
    'mmdet::_base_/models/mask-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_instance.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py',
]

# ---- Model: 1-class building head ----
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1),
    ),
    # COCO defaults assume bigger min/max; our tiles are 512x512.
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.5,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5,
        ),
    ),
)

# ---- Dataset ----
# Layout produced by scripts/coco_to_masks.py + the existing annotations/.
# All paths are relative to the project root (where you launch `python -m`).
data_root = ''  # use absolute / project-relative paths below
dataset_type = 'CocoDataset'
classes = ('building',)
metainfo = dict(classes=classes)

# Image tiles already 512x512; let mmdet just pad to multiples of 32.
image_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor')),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='building_extraction/annotations/vaihingen_train_coco.json',
        data_prefix=dict(img='building_extraction/data/train_images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=8),
        metainfo=metainfo,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='building_extraction/annotations/vaihingen_val_coco.json',
        data_prefix=dict(img='building_extraction/data/val_images/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='building_extraction/annotations/vaihingen_val_coco.json',
    metric=['bbox', 'segm'],
    format_only=False,
)
test_evaluator = val_evaluator

# ---- Schedule: linear-scaled for batch=8 on 1 GPU ----
# mmdet defaults to LR 0.02 for batch 16 (8 GPUs * 2). We're at batch 8 on 1 GPU
# (1/4 of default) so LR -> 0.005. Empirically safer than 0.01 with COCO init.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True,
         milestones=[8, 11], gamma=0.1),
]

# Save best checkpoint by mask mAP (the metric we actually care about for contour quality).
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/segm_mAP',
        rule='greater',
        max_keep_ckpts=3,
    ),
)

# Use COCO-pretrained Mask R-CNN as init.
load_from = ('https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/'
             'mask_rcnn_r50_fpn_2x_coco/'
             'mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_'
             '20200505_003907-3e542a40.pth')

work_dir = 'checkpoints/mmdet_vaihingen'
