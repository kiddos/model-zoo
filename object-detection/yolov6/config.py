import json

config = dict(model=dict(
  backbone_args=dict(
    in_channels=3,
    num_repeats=[1, 6, 12, 18, 6],
    out_channels=[64, 128, 256, 512, 1024],
    fuse_P2=True,
    cspsppf=True,
  ),
  neck_args=dict(
    num_repeats=[12, 12, 12, 12],
    out_channels=[256, 128, 128, 256, 256, 512],
  ),
  head_args=dict(
    in_channels=[128, 256, 512],
    num_layers=3,
    begin_indices=24,
    anchors=3,
    anchors_init=[[10, 13, 19, 19, 33, 23], [30, 61, 59, 59, 59, 119],
                  [116, 90, 185, 185, 373, 326]],
    out_indices=[17, 20, 23],
    strides=[8, 16, 32],
    atss_warmup_epoch=0,
    iou_type='siou',
    use_dfl=False,
    reg_max=0,
    distill_weight={
      'class': 1.0,
      'dfl': 1.0,
    },
  ),
  depth_multiple=0.33,
  width_multiple=0.25,
  num_classes=80,
), )


with open('yolov6n.json', 'w') as f:
  json.dump(config, f, indent=2)
