import torch


def dist2bbox(distance, anchor_points, box_format='xyxy'):
  '''Transform distance(ltrb) to box(xywh or xyxy).'''
  lt, rb = torch.split(distance, 2, -1)
  x1y1 = anchor_points - lt
  x2y2 = anchor_points + rb
  if box_format == 'xyxy':
    bbox = torch.cat([x1y1, x2y2], -1)
  elif box_format == 'xywh':
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    bbox = torch.cat([c_xy, wh], -1)
  return bbox

