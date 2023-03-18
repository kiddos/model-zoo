import torch.nn as nn
from torch.optim import SGD, Adam


def build_optimizer(config, model):
  """ Build optimizer from cfg file."""

  g_bnw, g_w, g_b = [], [], []
  for v in model.modules():
    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
      g_b.append(v.bias)
    if isinstance(v, nn.BatchNorm2d):
      g_bnw.append(v.weight)
    elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
      g_w.append(v.weight)

  solver = config['solver']
  optim = solver['optim']
  lr0 = solver['lr0']
  momentum = solver['momentum']
  weight_decay = solver['weight_decay']
  assert optim == 'SGD' or 'Adam', 'ERROR: unknown optimizer, use SGD defaulted'
  if optim == 'SGD':
    optimizer = SGD(g_bnw, lr=lr0, momentum=momentum, nesterov=True)
  elif optim == 'Adam':
    optimizer = Adam(g_bnw, lr=lr0, betas=(momentum, 0.999))

  optimizer.add_param_group({'params': g_w, 'weight_decay': weight_decay})
  optimizer.add_param_group({'params': g_b})

  del g_bnw, g_w, g_b
  return optimizer
