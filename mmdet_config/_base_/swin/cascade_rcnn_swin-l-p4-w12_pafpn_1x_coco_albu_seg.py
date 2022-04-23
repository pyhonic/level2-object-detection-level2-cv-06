_base_ = [
    '../models/cascade_rcnn_r50_fpn.py',
    '../datasets/coco_detection.py',
    '../schedules/cascade_rcnn_swin_l_schedule_1x.py',
    '../default_runtime.py'
]

pretrained = './swin_large_patch4_window12_384_22k.pth'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
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
        type='PAFPN',
        in_channels=[192, 384, 768, 1536]))


albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        p=0.3),
    dict(
        type='OneOf',
            transforms=[
                dict(
                    type='RandomBrightnessContrast',
                    p=1.0),
                dict(
                    type='CLAHE',
                    p=1.0),
                dict(
                    type='HueSaturationValue',
                    hue_shift_limit=20,
                    sat_shift_limit=50,
                    val_shift_limit=50,
                    p=1.0)],
        p=0.3),
    dict(
        type='OneOf',
            transforms=[
                    dict(type='Blur', blur_limit=7, p=0.5),
                    dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.5)
                    ],
        p=0.3),]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        keymap=dict(img="image", gt_bboxes="bboxes"),
        update_pad_shape=False,
        skip_img_without_anno=True,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True)
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]