_base_ = './cascade_rcnn_swin-t-p4-w7_fpn_1x_coco.py'
pretrained = '/opt/ml/detection/baseline/mmdetection/pth/swin_base_patch4_window12_384_22k.pth'  # noqa
model = dict(
    backbone=dict(
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4,8,16,32],
        window_size=12,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[128,256,512,1024]))