# TODO: Remove this config after benchmarking all related configs
_base_ = 'fcos.py'

data = dict(samples_per_gpu=4, workers_per_gpu=4)
