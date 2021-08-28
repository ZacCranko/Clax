from typing import Dict, Any
import jax, ml_collections, logging, time
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import numpy as jnp

from . import data

# number of channels per image
NUM_CH = 3

def _to_numpy(batch):
  return jax.tree_map(lambda x: x._numpy(), batch)

@jax.jit
def _concat_augs_along_batch(batch):
    
    aug_images, labels = batch
    num_augs = aug_images.shape[-1] // NUM_CH
    images = jnp.concatenate(jnp.split(aug_images, num_augs, axis = -1))
    labels = jnp.concatenate([labels for _ in range(num_augs)], axis = 0)

    return images, labels

@jax.jit
def _local_device_split(batch):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  
  @jax.jit 
  def _prepare(x):
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, batch)

def create_input_iter(config: ml_collections.ConfigDict, is_contrastive: bool, split: str = 'train', dataset = None, aug_shape: str = 'striped'):
  dataset_builder = tfds.builder(dataset if dataset is not None else config.dataset)
  ds = data.get_dataset(dataset_builder, batch_size = config.batch_size, is_contrastive = is_contrastive, cache_dataset = config.cache_dataset)

  ds = map(_to_numpy, ds)
  ds = map(_concat_augs_along_batch, ds)
  ds = map(_local_device_split, ds)

  dataset_iter = TrainIterator(dataset_builder = dataset_builder, dataset_iter = ds, batch_size = config.batch_size, 
                                      split = split, num_steps = config.num_steps, num_epochs = config.num_epochs, start_step = config.start_step)

  if split == 'test':
    dataset_iter.set_epochs(1)

  return dataset_iter

class TrainIterator:
  " ``There's no kill quite like overkill.'' "
  def __init__(self,  dataset_builder: tf.data.Dataset, dataset_iter, 
               batch_size: int, split: str = 'train', 
               num_steps: int = -1, num_epochs: int = -1, start_step: int = 0):
    
    if (num_steps <= 0 and num_epochs <= 0):
      raise ValueError("Must supply a positive number of either steps or epochs")


    self.dataset_builder = dataset_builder
    self.dataset_iter = dataset_iter.__iter__()
    self.batch_size = batch_size
    
    self.num_examples = dataset_builder.info.splits[split].num_examples
    self.image_shape  = dataset_builder.info.features['image'].shape
    self.num_classes  = dataset_builder.info.features['label'].num_classes
    self.shape = (self.image_shape, self.num_classes)

    self.steps_per_epoch = self.num_examples // self.batch_size 
    
    self.num_steps  = num_steps if num_steps > 0 else num_epochs * self.steps_per_epoch
    self.start_step = start_step 

    if self.start_step >= self.num_steps:
      raise ValueError(f"Must have start_step (got {self.start_step}) earlier than num_step (got {self.num_steps})")

    # force call of self.reset() before iterating
    self.global_step = num_steps
    self.train_start_time = float('inf')
    self.batch_start_time = float('inf')

    # time metrics
    self.batch_time = float('inf')
    self.images_per_second = float('inf')
    self.seconds_per_epoch = float('inf')

  def __len__(self):
    return self.num_steps - self.start_step

  def __iter__(self):
    self.__reset__()
    return self

  def __call__(self, *, info: str = None):
    if info is not None:
        logging.info(info)
    return self

  def __next__(self):
    if self.global_step == self.num_steps:
      raise StopIteration
    
    self._update_time_metrics()  
    self.global_step += 1
    
    return next(self.dataset_iter)

  def __reset__(self):
    self.train_start_time = time.time()
    self.batch_start_time = time.time()
    self.global_step = self.start_step - 1 

  def set_epochs(self, num_epochs: int):
    self.num_steps = num_epochs * self.steps_per_epoch

  def _update_time_metrics(self):
    self.batch_time = time.time() - self.batch_start_time
    self.batch_start_time = time.time()

    self.images_per_second = self.batch_size / self.batch_time
    self.seconds_per_epoch = self.steps_per_epoch * self.batch_time
 
  def is_freq(self, *, step_freq: int = -1, epoch_freq: int = -1, force_last: bool = False) -> bool:
    if force_last and (self.global_step + 1 == self.num_steps):
        return True 
    elif step_freq > 0:
      return (self.global_step + 1) % step_freq == 0
    elif epoch_freq > 0:
      return self.get_epoch(float = False) % epoch_freq == 0
    else:
       raise ValueError(f"Must supply positive step_freq (got {step_freq}) or epoch_freq (got {epoch_freq})")
  
  def get_metrics(self):
    metrics = {
      'epoch' : self.get_epoch(float = True),
      'perf'  : {
        'seconds_per_batch' : self.batch_time, 
        'images_per_second' : self.images_per_second,
        'seconds_per_epoch' : self.seconds_per_epoch
      }
    }
    return metrics
  
  def append_metrics(self, summary: Dict[str, Any], prefix: str = ""):
    for k, v in self.get_metrics().items():
      summary[f"{prefix}{k}"] = v

    return summary

  def is_epoch_start(self) -> bool: 
    if self.global_step == self.start_step:
      return True
    elif self.global_step == self.num_steps - 1:
      return False
    else: 
      return self.is_freq(step_freq = self.steps_per_epoch)

  def is_epoch_end(self) -> bool: 
    if self.global_step == self.num_steps - 1:
      return True 
    else:
      return self.is_freq(step_freq = self.steps_per_epoch)
  
  def is_train_start(self) -> bool:
    return self.global_step == self.start_step

  def get_epoch(self, *, float: bool = False) -> Any:
    if float:
      return (self.global_step + 1) / self.steps_per_epoch
    else:
      return (self.global_step + 1) // self.steps_per_epoch

