# Self-training round 1: Mask R-CNN warm-started from the Vaihingen source
# model, then trained on Vaihingen GT + Potsdam pseudo-labels mixed.
#
# Why mixed batches:
#   Pseudo-labels are noisy. Training on them alone causes the model to drift
#   toward whatever the source-model bias is on Potsdam (over-prediction on
#   impervious surfaces in our case). Keeping Vaihingen GT in every epoch
#   anchors the model to clean, real annotations.
#
# Why short schedule + low LR:
#   We're refining a model that already works on the source domain. Long
#   schedules amplify pseudo-label noise. A few epochs at 1/4 LR is the
#   self-training norm.
#
# Pre-flight: scripts/generate_pseudo_labels_mmdet.py must have produced
#   data/potsdam_pseudo_coco.json + data/potsdam_pseudo_images/.

_base_ = './mask_rcnn_vaihingen.py'

# ---- Source-domain dataset (unchanged from base) ----
# We re-declare it explicitly so we can put it inside ConcatDataset below.
_source_train = dict(
    type='CocoDataset',
    data_root='',
    ann_file='building_extraction/annotations/vaihingen_train_coco.json',
    data_prefix=dict(img='building_extraction/data/train_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=8),
    metainfo=dict(classes=('building',)),
    pipeline={{_base_.train_pipeline}},
)

# ---- Target-domain pseudo-label dataset ----
_target_train = dict(
    type='CocoDataset',
    data_root='',
    ann_file='building_extraction/data/potsdam_pseudo_coco.json',
    data_prefix=dict(img='building_extraction/data/potsdam_pseudo_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=8),
    metainfo=dict(classes=('building',)),
    pipeline={{_base_.train_pipeline}},
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[_source_train, _target_train],
    ),
)

# ---- Schedule: short, low LR, warm-started ----
# Source model used SGD lr=0.005. Self-training drops by 4x.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=200),
    dict(type='MultiStepLR', begin=0, end=6, by_epoch=True,
         milestones=[4, 5], gamma=0.1),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=1)

# Warm-start from the source-domain best checkpoint. Replace the path below
# if your saved best has a different name (mmdet appends the metric value).
load_from = 'checkpoints/mmdet_vaihingen/best_coco_segm_mAP.pth'

work_dir = 'checkpoints/mmdet_st_r1'
