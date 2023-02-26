_base_ = './mobileneck_resnet50.py'

model = dict(neck=dict(use_dcn=False))
