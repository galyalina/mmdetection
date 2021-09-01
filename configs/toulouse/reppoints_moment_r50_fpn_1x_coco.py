# The new config inherits a base config to highlight the necessary modification
_base_ = 'reppoints/reppoints_moment_r50_fpn_1x_coco.py'

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('building',)
data = dict(
    train=dict(
        img_prefix='data/toulouse/train/',
        classes=classes,
        ann_file='data/toulouse/annotations/instances_train.json'),
    val=dict(
        img_prefix='data/toulouse/val/',
        classes=classes,
        ann_file='data/toulouse/annotations/instances_val.json'),
    test=dict(
        img_prefix='data/toulouse/val/',
        classes=classes,
        ann_file='data/toulouse/annotations/instances_val.json'))