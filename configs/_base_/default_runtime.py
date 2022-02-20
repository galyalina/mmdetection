checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'DEBUG'
load_from = None
resume_from = None
# resume_from = 'work_dirs/reppoints_moment_r101_fpn_gn-neck+head_2x_toulouse/epoch_23.pth'
workflow = [('train', 1)]
