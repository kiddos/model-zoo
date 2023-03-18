import math
import logging
import torch


def build_lr_scheduler(config, optimizer, epochs):
  """Build learning rate scheduler from cfg file."""
  solver = config['solver']
  lr_scheduler = solver['lr_scheduler']
  lrf = solver['lrf']

  if lr_scheduler == 'Cosine':
    lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1
  elif lr_scheduler == 'Constant':
    lf = lambda x: 1.0
  else:
    logging.error('unknown lr scheduler, use Cosine defaulted')

  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
  return scheduler, lf
