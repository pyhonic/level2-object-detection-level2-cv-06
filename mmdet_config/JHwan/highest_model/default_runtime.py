checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',interval=10,
            init_kwargs=dict(
                project='JHwan96',
                entity = 'omakase06',
                name = 'faster rcnn swin multiscale albu'
            ),
            )        
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
fp16=None
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
