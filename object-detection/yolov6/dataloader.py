import os
import os.path as path
import json

from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

from dataset import COCODataset


class COCODataLoader(DataLoader):
  """Dataloader that reuses workers
  Uses same syntax as vanilla DataLoader
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
    self.iterator = super().__iter__()

  def __len__(self):
    return len(self.batch_sampler.sampler)

  def __iter__(self):
    for i in range(len(self)):
      yield next(self.iterator)


class _RepeatSampler:
  """Sampler that repeats forever
  Args:
      sampler (Sampler)
  """

  def __init__(self, sampler):
    self.sampler = sampler

  def __iter__(self):
    while True:
      yield from iter(self.sampler)


def get_dataloader(args, config):
  home = os.getenv('HOME')
  coco_root = path.join(home, 'ssd_datasets', 'coco2017')
  train_dataset = CocoDetection(
    path.join(coco_root, 'images', 'train2017'),
    path.join(coco_root, 'annotations', 'instances_train2017.json')
  )
  val_dataset = CocoDetection(
    path.join(coco_root, 'images', 'val2017'),
    path.join(coco_root, 'annotations', 'instances_val2017.json')
  )

  with open('coco_categories.json', 'r') as f:
    category_mapping = json.load(f)

  img_size = args.img_size
  data_aug = config['data_aug']
  train_coco_dataset = COCODataset(
    train_dataset,
    category_mapping,
    img_size=img_size,
    augment=True,
    hyp=dict(data_aug),
    task='train'
  )
  val_coco_dataset = COCODataset(
    val_dataset, category_mapping, img_size=img_size, hyp=dict(data_aug), task='val'
  )

  batch_size = args.batch_size
  num_workers = args.workers
  train_dataloader = COCODataLoader(
    train_coco_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=COCODataset.collate_fn,
  )

  val_dataloader = COCODataLoader(
    val_coco_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=COCODataset.collate_fn,
  )
  return train_dataloader, val_dataloader, len(category_mapping)
