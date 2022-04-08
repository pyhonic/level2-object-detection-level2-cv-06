checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        
        dict(type='WandbLoggerHook',
            interval=1000,
            init_kwargs=dict(
            entity = 'omakase06', # 설정 따로 안 해도 됨
            project='PDH', # 이름
            name = 'cascade_swin_800' # 실험제목
            ))
    ],
)
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
