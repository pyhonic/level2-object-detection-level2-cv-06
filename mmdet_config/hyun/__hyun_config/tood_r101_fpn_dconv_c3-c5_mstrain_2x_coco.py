_base_ = './tood_r101_fpn_mstrain_2x_coco.py'


pretrained = '/opt/ml/detection/baseline/mmdetection/configs/__hyun_config/pth/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20211210_213728-4a824142.pth'

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    bbox_head=dict(num_dcn=2))
