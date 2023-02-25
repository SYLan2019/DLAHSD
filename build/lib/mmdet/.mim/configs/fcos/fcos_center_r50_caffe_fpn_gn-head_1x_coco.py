_base_ = './fcos.py'
model = dict(bbox_head=dict(center_sampling=True, center_sample_radius=1.5))
