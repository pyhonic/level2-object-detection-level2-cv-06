# 모듈 import
import os
import gc
import torch
import wandb
import argparse

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)

def main(args):
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    
    base = args.base_path
    root = args.data_dir
    config_file = args.config_file
    n_fold = args.n_fold
    is_resume = True if args.resume_from else False

    name = f'{config_file.split(".")[0]}_{"no_fold" if n_fold == 0 else "fold_" + str(n_fold)}'
    kfold = "" if n_fold == 0 else 'stratified_kfold'
    kfold_path = os.path.join(root, kfold)

    cfg = Config.fromfile(os.path.join(base, config_file))
    if is_resume:
        cfg.resume_from = args.resume_from

    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = os.path.join(kfold_path, f'train{"" if not n_fold else "_" + str(n_fold)}.json')

    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = os.path.join(kfold_path, f'val{"" if not n_fold else "_" + str(n_fold)}.json')

    cfg.data.samples_per_gpu = 2

    cfg.seed = 2022
    cfg.gpu_ids = [0]
    cfg.work_dir = f'./work_dirs/{config_file.split(".")[0]}_{"no_fold" if n_fold == 0 else "fold_" + str(n_fold)}'

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=1, interval=1)

    cfg.log_config.hooks[1].exp_name = name
    
    gc.collect()
    torch.cuda.empty_cache()

    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model)
    model.init_weights()

    train_detector(model, datasets[0], cfg, distributed=False, validate=n_fold)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_path', type=str, default='/opt/ml/detection/baseline/mmdetection/configs/_ho_/_base_/swin'
    )
    parser.add_argument(
        '--data_dir', type=str, default='/opt/ml/detection/dataset/'
    )
    parser.add_argument(
        '--config_file', type=str, default='cascade_rcnn_swin-l-p4-w12_pafpn_1x_coco_albu.py'
    )
    parser.add_argument(
        '--n_fold', type=int, default=1,
    )
    parser.add_argument(
        '--resume_from', type=str, default='',
    )
    
    args = parser.parse_args()
    main(args)