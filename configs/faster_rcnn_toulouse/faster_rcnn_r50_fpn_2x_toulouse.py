_base_ = [
    './faster_rcnn_r50_fpn_toulouse.py',
    './toulouse_detection_augmented.py',
    '../_base_/schedules/schedule_2x.py', './default_runtime.py'
]
