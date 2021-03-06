from . import data_util

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
      test_crop=test_crop,
  )


def get_dataset(
    builder: tfds.core.DatasetBuilder,
    *,
    batch_size: int,
    shuffle_files: bool = True,
    is_pretrain: bool = True,
    split: str = "train",
    cache_dataset: bool = True,
    num_transforms: int = 2,
):

  num_classes = builder.info.features["label"].num_classes
  image_size, _, _ = builder.info.features["image"].shape

  preprocess_fn = get_preprocess_fn(shuffle_files,
                                    image_size,
                                    is_pretrain=is_pretrain)

  def map_fn(image, label):
    """Produces multiple transformations of the same batch."""
    if num_transforms != -1:
      xs = []
      for _ in range(num_transforms):
        # Two transformations
        xs.append(preprocess_fn(image))

      image = tf.concat(xs, -1)

    label = tf.one_hot(label, num_classes)
    return image, label

  dataset = builder.as_dataset(
      split=split,
      shuffle_files=shuffle_files,
      as_supervised=True,
      # Passing the input_context to TFDS makes TFDS read different parts
      # of the dataset on different workers. We also adjust the interleave
      # parameters to achieve better performance.
      read_config=tfds.ReadConfig(interleave_cycle_length=32,
                                  interleave_block_length=1,
                                  input_context=None),
  )
  if cache_dataset:
    dataset = dataset.cache()
  if shuffle_files:
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_slack = True
    dataset = dataset.with_options(options)
    buffer_multiplier = 50 if image_size <= 32 else 10
    dataset = dataset.shuffle(batch_size * buffer_multiplier)

  dataset = dataset.repeat(-1)
  dataset = dataset.map(map_fn,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=shuffle_files)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset
