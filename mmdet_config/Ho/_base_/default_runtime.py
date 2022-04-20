checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            interval=1000,
            init_kwargs=dict(
                project='default', 
                name = 'exp' 
            ))
        ])


custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

opencv_num_threads = 0
mp_start_method = 'fork'