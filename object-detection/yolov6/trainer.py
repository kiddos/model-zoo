import os
import os.path as path
import time
from copy import deepcopy
import logging
import shutil
from tqdm import tqdm
import numpy as np
import torch
from torch.cuda import amp

from dataloader import get_dataloader
from model import YOLOv6
from ema import ModelEMA
from optimizer import build_optimizer
from scheduler import build_lr_scheduler
from loss import ComputeLoss, ComputeLossAnchorBase

NCOLS = min(100, shutil.get_terminal_size().columns)


class Trainer(object):

  def __init__(self, args, config, device):
    self.args = args
    self.config = config
    self.device = device

    # get data loader
    self.train_loader, self.val_loader, self.num_classes = get_dataloader(args, config)

    #  # get model and optimizer
    self.model = YOLOv6(config, 80, args.fuse_ab, args.distill)
    self.model.to(self.device)

    self.optimizer = self.get_optimizer(args, config, self.model)
    self.scheduler, self.lf = self.get_lr_scheduler(args, config, self.optimizer)
    self.ema = ModelEMA(self.model)

    self.start_epoch = 0

    self.max_epoch = args.epochs
    self.max_stepnum = len(self.train_loader)
    self.batch_size = args.batch_size
    self.img_size = args.img_size
    self.save_dir = path.join(args.output_dir, self.config['model']['name'] + '.pt')

    self.loss_num = 3
    self.loss_info = ['Epoch', 'iou_loss', 'dfl_loss', 'cls_loss']

  def get_optimizer(self, args, config, model):
    accumulate = max(1, round(64 / args.batch_size))
    solver = config['solver']
    solver['weight_decay'] *= args.batch_size * accumulate / 64
    optimizer = build_optimizer(config, model)
    return optimizer

  def get_lr_scheduler(self, args, config, optimizer):
    epochs = args.epochs
    lr_scheduler, lf = build_lr_scheduler(config, optimizer, epochs)
    return lr_scheduler, lf

  # Training Process
  def train(self):
    try:
      self.train_before_loop()
      for self.epoch in range(self.start_epoch, self.max_epoch):
        self.train_in_loop(self.epoch)

    except Exception:
      logging.error('ERROR in training loop or eval/save model.')
      raise
    finally:
      self.train_after_loop()

  def train_before_loop(self):
    logging.info('Training start...')
    self.start_time = time.time()

    solver = self.config['solver']
    warmup_epochs = solver['warmup_epochs']
    self.warmup_stepnum = max(round(warmup_epochs * self.max_stepnum), 1000)
    self.scheduler.last_epoch = self.start_epoch - 1
    self.last_opt_step = -1
    self.scaler = amp.GradScaler(enabled=self.device != 'cpu')

    self.best_ap, self.ap = 0.0, 0.0
    self.best_stop_strong_aug_ap = 0.0
    self.evaluate_results = (0, 0)  # AP50, AP50_95

    model_conf = self.config['model']
    head_conf = model_conf['head']
    atss_warmup_epoch = head_conf['atss_warmup_epoch']
    use_dfl = head_conf['use_dfl']
    reg_max = head_conf['reg_max']
    iou_type = head_conf['iou_type']
    strides = head_conf['strides']
    self.compute_loss = ComputeLoss(
      num_classes=self.num_classes,
      ori_img_size=self.img_size,
      warmup_epoch=atss_warmup_epoch,
      use_dfl=use_dfl,
      reg_max=reg_max,
      iou_type=iou_type,
      fpn_strides=strides
    )

    if self.args.fuse_ab:
      self.compute_loss_ab = ComputeLossAnchorBase(
        num_classes=self.num_classes,
        ori_img_size=self.img_size,
        warmup_epoch=0,
        use_dfl=False,
        reg_max=0,
        iou_type=iou_type,
        fpn_strides=strides
      )

  # Training loop for each epoch
  def train_in_loop(self, epoch_num):
    try:
      self.prepare_for_steps()
      for self.step, self.batch_data in self.pbar:
        self.train_in_steps(epoch_num, self.step)
        self.print_details()
    except Exception:
      logging.error('ERROR in training steps.')
      raise

    try:
      self.eval_and_save()
    except Exception:
      logging.error('ERROR in evaluate and save model.')
      raise

  def prepare_for_steps(self):
    if self.epoch > self.start_epoch:
      self.scheduler.step()

    # stop strong aug like mosaic and mixup from last n epoch by recreate dataloader
    if self.epoch == self.max_epoch - self.args.stop_aug_last_n_epoch:
      self.cfg.data_aug.mosaic = 0.0
      self.cfg.data_aug.mixup = 0.0
      self.train_loader, self.val_loader = self.get_data_loader(self.args, self.cfg, self.data_dict)

    self.model.train()
    self.mean_loss = torch.zeros(self.loss_num, device=self.device)
    self.optimizer.zero_grad()

    logging.info(('\n' + '%10s' * (self.loss_num + 1)) % (*self.loss_info,))
    self.pbar = enumerate(self.train_loader)
    self.pbar = tqdm(
      self.pbar, total=self.max_stepnum, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
    )

  # Training loop for batchdata
  def train_in_steps(self, epoch_num, step_num):
    images, targets = self.prepro_data(self.batch_data, self.device)

    # forward
    with amp.autocast(enabled=self.device != 'cpu'):
      preds, s_featmaps = self.model(images)
      if self.args.fuse_ab:
        total_loss, loss_items = self.compute_loss(
          (preds[0], preds[3], preds[4]), targets, epoch_num, step_num
        )  # YOLOv6_af
        total_loss_ab, loss_items_ab = self.compute_loss_ab(
          preds[:3], targets, epoch_num, step_num
        )  # YOLOv6_ab
        total_loss += total_loss_ab
        loss_items += loss_items_ab
      else:
        total_loss, loss_items = self.compute_loss(preds, targets, epoch_num, step_num)  # YOLOv6_af

    # backward
    self.scaler.scale(total_loss).backward()
    self.loss_items = loss_items
    self.update_optimizer()

  def prepro_data(self, batch_data, device):
    images = batch_data[0].to(device, non_blocking=True).float() / 255
    targets = batch_data[1].to(device)
    return images, targets

  def update_optimizer(self):
    curr_step = self.step + self.max_stepnum * self.epoch
    self.accumulate = max(1, round(64 / self.batch_size))
    if curr_step <= self.warmup_stepnum:
      self.accumulate = max(
        1,
        np.interp(curr_step, [0, self.warmup_stepnum], [1, 64 / self.batch_size]).round()
      )
      for k, param in enumerate(self.optimizer.param_groups):
        solver = self.config['solver']
        warmup_bias_lr = solver['warmup_bias_lr']
        warmup_bias_lr = warmup_bias_lr if k == 2 else 0.0
        param['lr'] = np.interp(
          curr_step, [0, self.warmup_stepnum],
          [warmup_bias_lr, param['initial_lr'] * self.lf(self.epoch)]
        )
        if 'momentum' in param:
          warmup_momentum = solver['warmup_momentum']
          momentum = solver['momentum']
          param['momentum'] = np.interp(
            curr_step, [0, self.warmup_stepnum], [warmup_momentum, momentum]
          )
    if curr_step - self.last_opt_step >= self.accumulate:
      self.scaler.step(self.optimizer)
      self.scaler.update()
      self.optimizer.zero_grad()
      if self.ema:
        self.ema.update(self.model)
      self.last_opt_step = curr_step

  # Print loss after each steps
  def print_details(self):
    self.mean_loss = (self.mean_loss * self.step + self.loss_items) / (self.step + 1)
    self.pbar.set_description(
      ('%10s' + '%10.4g' * self.loss_num) %
      (f'{self.epoch}/{self.max_epoch - 1}', *(self.mean_loss))
    )

  # Empty cache if training finished
  def train_after_loop(self):
    if self.device != 'cpu':
      torch.cuda.empty_cache()

  def eval_and_save(self):
    remaining_epochs = self.max_epoch - self.epoch
    eval_interval = self.args.eval_interval if remaining_epochs > self.args.heavy_eval_range else 3
    is_val_epoch = (not self.args.eval_final_only or
                    (remaining_epochs == 1)) and (self.epoch % eval_interval == 0)

    self.ema.update_attr(
      self.model, include=['nc', 'names', 'stride']
    )  # update attributes for ema model

    # TODO
    #  if is_val_epoch:
    #    self.eval_model()
    #    self.ap = self.evaluate_results[1]
    #    self.best_ap = max(self.ap, self.best_ap)
    # save ckpt
    ckpt = {
      'model': deepcopy(self.model).half(),
      'ema': deepcopy(self.ema.ema).half(),
      'updates': self.ema.updates,
      'optimizer': self.optimizer.state_dict(),
      'epoch': self.epoch,
    }

    save_ckpt_dir = path.join(self.save_dir, 'weights')
    self.save_checkpoint(
      ckpt, (is_val_epoch) and (self.ap == self.best_ap), save_ckpt_dir, model_name='last_ckpt'
    )
    if self.epoch >= self.max_epoch - self.args.save_ckpt_on_last_n_epoch:
      self.save_checkpoint(ckpt, False, save_ckpt_dir, model_name=f'{self.epoch}_ckpt')

    # default save best ap ckpt in stop strong aug epochs
    if self.epoch >= self.max_epoch - self.args.stop_aug_last_n_epoch:
      if self.best_stop_strong_aug_ap < self.ap:
        self.best_stop_strong_aug_ap = max(self.ap, self.best_stop_strong_aug_ap)
        self.save_checkpoint(ckpt, False, save_ckpt_dir, model_name='best_stop_aug_ckpt')

    del ckpt
    # log for learning rate
    lr = [x['lr'] for x in self.optimizer.param_groups]
    self.evaluate_results = list(self.evaluate_results) + lr

  def save_checkpoint(self, ckpt, is_best, save_dir, model_name=""):
    """ Save checkpoint to the disk."""
    if not path.exists(save_dir):
      os.makedirs(save_dir)

    filename = path.join(save_dir, model_name + '.pt')
    torch.save(ckpt, filename)
    if is_best:
      best_filename = path.join(save_dir, 'best_ckpt.pt')
      shutil.copyfile(filename, best_filename)
