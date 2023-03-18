import argparse
import logging
import json

import torch
from trainer import Trainer


def get_args_parser(add_help=True):
  parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Training', add_help=add_help)

  parser.add_argument('--img-size', default=640, type=int, help='train, val image size (pixels)')
  parser.add_argument('--batch-size', default=20, type=int, help='total batch size for all GPUs')
  parser.add_argument('--epochs', default=400, type=int, help='number of total epochs to run')
  parser.add_argument('--config', default='yolov6n.json', help='config to load')
  parser.add_argument(
    '--workers', default=8, type=int, help='number of data loading workers (default: 8)'
  )
  parser.add_argument(
    '--fuse_ab', action='store_true', default=True, help='fuse ab branch in training process or not'
  )

  parser.add_argument('--output-dir', default='./runs/train', type=str, help='path to save outputs')
  parser.add_argument(
    '--eval-final-only', action='store_true', help='only evaluate at the final epoch'
  )
  parser.add_argument(
    '--eval-interval', default=20, type=int, help='evaluate at every interval epochs'
  )
  parser.add_argument(
    '--heavy-eval-range',
    default=50,
    type=int,
    help='evaluating every epoch for last such epochs (can be jointly used with --eval-interval)'
  )
  parser.add_argument(
    '--stop_aug_last_n_epoch',
    default=15,
    type=int,
    help='stop strong aug at last n epoch, neg value not stop, default 15'
  )


  parser.add_argument(
    '--resume', nargs='?', const=True, default=False, help='resume the most recent training'
  )
  parser.add_argument(
    '--save_ckpt_on_last_n_epoch',
    default=-1,
    type=int,
    help='save last n epoch even not best or last, neg value not save'
  )

  parser.add_argument('--distill', action='store_true', help='distill or not')
  parser.add_argument('--distill_feat', action='store_true', help='distill featmap or not')
  parser.add_argument('--quant', action='store_true', help='quant or not')
  parser.add_argument('--calib', action='store_true', help='run ptq')
  parser.add_argument('--teacher_model_path', type=str, default=None, help='teacher model path')
  parser.add_argument('--temperature', type=int, default=20, help='distill temperature')
  return parser


def load_config(target):
  with open(target, 'r') as f:
    return json.load(f)


def main():
  args = get_args_parser().parse_args()
  # reload envs because args was chagned in check_and_init(args)
  logging.info(f'training args are: {args}\n')

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  config = load_config(args.config)
  # Start
  trainer = Trainer(args, config, device)
  trainer.train()


if __name__ == '__main__':
  main()
