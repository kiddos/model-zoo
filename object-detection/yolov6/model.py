import math
import torch
import torch.nn as nn

from backbone import EfficientRep
from neck import RepBiFPANNeck
from head import build_effidehead_layer_fuse_ab, DetectAnchorBase


def make_divisible(x, divisor):
  # Upward revision the value x to make it evenly divisible by the divisor.
  return math.ceil(x / divisor) * divisor


def initialize_weights(model):
  for m in model.modules():
    t = type(m)
    if t is nn.Conv2d:
      pass
    elif t is nn.BatchNorm2d:
      m.eps = 1e-3
      m.momentum = 0.03
    elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
      m.inplace = True


class YOLOv6(nn.Module):
  def __init__(self, config, num_classes, fuse_ab=False, distill_ns=False, input_channels=3):
    super().__init__()

    model_conf = config['model']
    depth_mul = model_conf['depth_multiple']
    width_mul = model_conf['width_multiple']

    backbone_conf = model_conf['backbone']
    num_repeat_backbone = backbone_conf['num_repeats']
    channels_list_backbone = backbone_conf['out_channels']
    fuse_P2 = backbone_conf['fuse_P2']
    cspsppf = backbone_conf['cspsppf']

    neck_conf = model_conf['neck']
    num_repeat_neck = neck_conf['num_repeats']
    channels_list_neck = neck_conf['out_channels']

    head_conf = model_conf['head']
    use_dfl = head_conf['use_dfl']
    reg_max = head_conf['reg_max']
    num_layers = head_conf['num_layers']
    anchors_init = head_conf['anchors_init']

    num_repeat = [
      (max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)
    ]
    channels_list = [
      make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)
    ]

    BACKBONE = None
    if backbone_conf['type'] == 'EfficientRep':
      BACKBONE = EfficientRep

    assert BACKBONE is not None, f'Backbone type {backbone_conf["type"]} not implemented'

    self.backbone = BACKBONE(
      in_channels=input_channels,
      channels_list=channels_list,
      num_repeats=num_repeat,
      fuse_P2=fuse_P2,
      cspsppf=cspsppf
    )

    NECK = None
    if neck_conf['type'] == 'RepBiFPANNeck':
      NECK = RepBiFPANNeck

    assert NECK is not None, f'Neck type {neck_conf["type"]} not implemented'

    self.neck = NECK(channels_list=channels_list, num_repeats=num_repeat)

    if fuse_ab:
      head_layers = build_effidehead_layer_fuse_ab(
        channels_list, 3, num_classes, reg_max=reg_max, num_layers=num_layers
      )
      self.detect = DetectAnchorBase(num_classes, anchors_init, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    # Init Detect head
    self.stride = self.detect.stride
    self.detect.initialize_biases()

    # Init weights
    initialize_weights(self)

  def forward(self, x):
    export_mode = torch.onnx.is_in_onnx_export()
    x = self.backbone(x)
    x = self.neck(x)
    if not export_mode:
      featmaps = []
      featmaps.extend(x)
    x = self.detect(x)
    return x if export_mode is True else [x, featmaps]

  def _apply(self, fn):
    self = super()._apply(fn)
    self.detect.stride = fn(self.detect.stride)
    self.detect.grid = list(map(fn, self.detect.grid))
    return self
