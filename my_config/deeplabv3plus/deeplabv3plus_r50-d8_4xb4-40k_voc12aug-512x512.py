_base_ = [
    './deeplabv3plus_r50-d8.py',
    './crack_dataset.py', './default_runtime.py',
    './schedule_20k.py'
]
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))
