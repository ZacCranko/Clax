#%%
from . import data_util
# import data_util
from tensorflow.python.data.ops.dataset_ops import Dataset
import tensorflow.compat.v2 as tf

import tensorflow_datasets as tfds
import functools
import time 
from absl import logging

def build_distributed_dataset(builder, batch_size, is_training, strategy,
                              topology):
  input_fn = build_input_fn(builder, batch_size, topology, is_training)
  return strategy.distribute_datasets_from_function(input_fn)


def get_preprocess_fn(is_training, image_size, is_pretrain):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  if image_size <= 32:
    test_crop = False
  else:
    test_crop = True
  return functools.partial(
      data_util.preprocess_image,
      height=image_size,
      width=image_size,
      is_training=is_training,
      color_distort=is_pretrain,
      test_crop=test_crop)


# def build_input_fn(builder, global_batch_size, topology, is_training):
#   """Build input function.

#   Args:
#     builder: TFDS builder for specified dataset.
#     global_batch_size: Global batch size.
#     topology: An instance of `tf.tpu.experimental.Topology` or None.
#     is_training: Whether to build in training mode.

#   Returns:
#     A function that accepts a dict of params and returns a tuple of images and
#     features, to be used as the input_fn in TPUEstimator.
#   """

#   def _input_fn(input_context):
#     """Inner input function."""
#     batch_size = input_context.get_per_replica_batch_size(global_batch_size)
#     logging.info('Global batch size: %d', global_batch_size)
#     logging.info('Per-replica batch size: %d', batch_size)
#     preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
#     preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)
#     num_classes = builder.info.features['label'].num_classes

#     def map_fn(image, label):
#       """Produces multiple transformations of the same batch."""
#       if is_training and FLAGS.train_mode == 'pretrain':
#         xs = []
#         for _ in range(2):  # Two transformations
#           xs.append(preprocess_fn_pretrain(image))
#         image = tf.concat(xs, -1)
#       else:
#         image = preprocess_fn_finetune(image)
#       label = tf.one_hot(label, num_classes)
#       return image, label

#     logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
#     dataset = builder.as_dataset(
#         split=FLAGS.train_split if is_training else FLAGS.eval_split,
#         shuffle_files=is_training,
#         as_supervised=True,
#         # Passing the input_context to TFDS makes TFDS read different parts
#         # of the dataset on different workers. We also adjust the interleave
#         # parameters to achieve better performance.
#         read_config=tfds.ReadConfig(
#             interleave_cycle_length=32,
#             interleave_block_length=1,
#             input_context=input_context))
#     if FLAGS.cache_dataset:
#       dataset = dataset.cache()
#     if is_training:
#       options = tf.data.Options()
#       options.experimental_deterministic = False
#       options.experimental_slack = True
#       dataset = dataset.with_options(options)
#       buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
#       dataset = dataset.shuffle(batch_size * buffer_multiplier)
#       dataset = dataset.repeat(-1)
#     dataset = dataset.map(
#         map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset = dataset.batch(batch_size, drop_remainder=is_training)
#     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
#     return dataset

#   return _input_fn
  
def get_dataset(builder: tfds.core.DatasetBuilder, 
                   batch_size: int, 
                   is_training: bool = True,  
                   train_mode: str = 'pretrain', 
                   split: str = 'train',
                   cache_dataset: bool = True):

  num_classes = builder.info.features['label'].num_classes
  image_size, _ , _ = builder.info.features['image'].shape
  
  preprocess_fn_pretrain = get_preprocess_fn(is_training, image_size, is_pretrain=True)
  preprocess_fn_finetune = get_preprocess_fn(is_training, image_size, is_pretrain=False)

  def map_fn(image, label):
    """Produces multiple transformations of the same batch."""
    if is_training and train_mode == 'pretrain':
      xs = []
      for _ in range(2):  # Two transformations
        xs.append(preprocess_fn_pretrain(image))
      # xs = [image, image]
      image = tf.concat(xs, -1)
    else:
      image = preprocess_fn_finetune(image)
    label = tf.one_hot(label, num_classes)
    return image, label

  dataset = builder.as_dataset(
      split = split,
      shuffle_files = is_training,
      as_supervised = True,
      # Passing the input_context to TFDS makes TFDS read different parts
      # of the dataset on different workers. We also adjust the interleave
      # parameters to achieve better performance.
      read_config=tfds.ReadConfig(
          interleave_cycle_length=32,
          interleave_block_length=1,
          input_context=None))
  if cache_dataset:
    dataset = dataset.cache()
  if is_training:
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    buffer_multiplier = 50 if image_size <= 32 else 10
    dataset = dataset.shuffle(batch_size * buffer_multiplier)

  dataset = dataset.repeat(-1)
  dataset = dataset.map(
      map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=is_training)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  
  return dataset

class DatasetIterator:
  def __init__(self,  dataset_builder, dataset_iter, 
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

    self.num_steps  = num_steps if num_steps >0 else num_epochs * self.steps_per_epoch
    self.num_epochs = self.num_steps // self.steps_per_epoch
    
    self.start_step = start_step 

    if self.start_step >= self.num_steps:
      raise ValueError(f"Must have start_step (got {self.start_step}) earlier than num_step (got {self.num_steps})")

    # force call of self.reset() before iterating
    self.global_step = num_steps
    self.train_start_time = float('inf')
    self.batch_start_time = float('inf')

    # time metrics
    self.batch_time = float('inf')
    self.samples_per_second = float('inf')
    self.seconds_per_epoch = float('inf')

  def __len__(self):
    return self.num_steps - self.start_step

  def __iter__(self):
    self._reset()
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

  def _reset(self):
    self.train_start_time = time.time()
    self.batch_start_time = time.time()
    self.global_step = self.start_step - 1 

  def _update_time_metrics(self):
    self.batch_time = time.time() - self.batch_start_time
    self.batch_start_time = time.time()

    self.samples_per_second = self.batch_size / self.batch_time
    self.seconds_per_epoch = self.steps_per_epoch * self.batch_time

  def get_time_metrics(self):
    metrics = {
      'batch_time ': self.batch_time, 
      'batch_samples_per_second' : self.samples_per_second,
      'batch_seconds_per_epoch' : self.seconds_per_epoch
    }
    return metrics
  
  def append_metrics(self, summary, prefix: str = ""):
    summary[f"{prefix}epoch"] = self.get_epoch(float = True)
    summary[f"{prefix}global_step"] = self.global_step

    for k, v in self.get_time_metrics().items():
      summary[f"{prefix}{k}"] = v

    return summary

  def is_epoch_start(self) -> bool: 
    return self.is_freq(step_freq = self.steps_per_epoch)
  
  def is_train_start(self) -> bool:
    return self.global_step == self.start_step

  def get_epoch(self, *, float: bool = False) -> int:
    if float:
      return (self.global_step + 1) / self.steps_per_epoch
    else:
      return (self.global_step + 1) // self.steps_per_epoch


  def is_freq(self, *, step_freq: int = -1, epoch_freq: int = -1, force_last: bool = False) -> bool:
    if force_last and (self.global_step + 1 == self.num_steps):
        return True 
    elif step_freq > 0:
      return self.global_step % step_freq == 0
    elif epoch_freq > 0:
      return self.get_epoch(float = False) % epoch_freq == 0
    else:
       raise ValueError(f"Must supply positive step_freq (got {step_freq}) or epoch_freq (got {epoch_freq})")