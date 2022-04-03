_base_ = [
    '_base_/models/faster_rcnn_r50_fpn_swin_l_ms.py',
    '_base_/datasets/coco_detection_ms_valid.py',
    '_base_/schedules/schedule_1x_cosine.py', '_base_/default_runtime.py'
]

pretrained='/opt/ml/detection/baseline/mmdetection/pth/swin_large_patch4_window12_384_22k.pth'  # noqa

model = dict(
    type='FasterRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6,12,24,48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='BIFPN',
        in_channels=[192,384,768,1536]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='CLAHE', clip_limit=2),
            dict(type='RandomBrightnessContrast'),
        ],
        p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=0.5),
            dict(type='MedianBlur', blur_limit=3, p=0.5)
        ],
        p=0.5),

]

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(512, 512), (544, 544), (576, 576), (608, 608),
                           (640, 640), (672, 672), (704, 704), (736, 736),
                           (768, 768), (800, 800), (832, 832), (864, 864),
                           (896, 896), (928, 928), (960, 960), (992, 992),
                           (1024, 1024), (1056, 1056), (1088, 1088),
                           (1120, 1120), (1152, 1152), (1184, 1184),
                           (1216, 1216), (1248, 1248), (1280, 1280),
                           (1312, 1312), (1344, 1344), (1376, 1376),
                           (1408, 1408), (1440, 1440), (1472, 1472),
                           (1504, 1504), (1536, 1536)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
#lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=15)

# (512, 512),  (576, 576), (608, 608),
#                             (672, 672), (768, 768), (864, 864), (992, 992),
#                             (1024, 1024), (1088, 1088), (1120, 1120), (1280, 1280),
#                             (1344, 1344)