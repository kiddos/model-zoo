import torch
import torch.nn as nn

from layers import RepVGGBlock, SimConv, BiFusion, RepBlock


class RepBiFPANNeck(nn.Module):
  """
  RepBiFPANNeck Module
  """

  # [64, 128, 256, 512, 1024]
  # [256, 128, 128, 256, 256, 512]

  def __init__(self, channels_list=None, num_repeats=None, block=RepVGGBlock):
    super().__init__()

    assert channels_list is not None
    assert num_repeats is not None

    self.reduce_layer0 = SimConv(
      in_channels=channels_list[4],  # 1024
      out_channels=channels_list[5],  # 256
      kernel_size=1,
      stride=1
    )

    self.Bifusion0 = BiFusion(
      in_channels=[channels_list[3], channels_list[5]],  # 512, 256
      out_channels=channels_list[5],  # 256
    )
    self.Rep_p4 = RepBlock(
      in_channels=channels_list[5],  # 256
      out_channels=channels_list[5],  # 256
      n=num_repeats[5],
      block=block
    )

    self.reduce_layer1 = SimConv(
      in_channels=channels_list[5],  # 256
      out_channels=channels_list[6],  # 128
      kernel_size=1,
      stride=1
    )

    self.Bifusion1 = BiFusion(
      in_channels=[channels_list[5], channels_list[6]],  # 256, 128
      out_channels=channels_list[6],  # 128
    )

    self.Rep_p3 = RepBlock(
      in_channels=channels_list[6],  # 128
      out_channels=channels_list[6],  # 128
      n=num_repeats[6],
      block=block
    )

    self.downsample2 = SimConv(
      in_channels=channels_list[6],  # 128
      out_channels=channels_list[7],  # 128
      kernel_size=3,
      stride=2
    )

    self.Rep_n3 = RepBlock(
      in_channels=channels_list[6] + channels_list[7],  # 128 + 128
      out_channels=channels_list[8],  # 256
      n=num_repeats[7],
      block=block
    )

    self.downsample1 = SimConv(
      in_channels=channels_list[8],  # 256
      out_channels=channels_list[9],  # 256
      kernel_size=3,
      stride=2
    )

    self.Rep_n4 = RepBlock(
      in_channels=channels_list[5] + channels_list[9],  # 256 + 256
      out_channels=channels_list[10],  # 512
      n=num_repeats[8],
      block=block
    )

  def forward(self, input):

    (x3, x2, x1, x0) = input

    fpn_out0 = self.reduce_layer0(x0)
    f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
    f_out0 = self.Rep_p4(f_concat_layer0)

    fpn_out1 = self.reduce_layer1(f_out0)
    f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
    pan_out2 = self.Rep_p3(f_concat_layer1)

    down_feat1 = self.downsample2(pan_out2)
    p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
    pan_out1 = self.Rep_n3(p_concat_layer1)

    down_feat0 = self.downsample1(pan_out1)
    p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
    pan_out0 = self.Rep_n4(p_concat_layer2)

    outputs = [pan_out2, pan_out1, pan_out0]

    return outputs
