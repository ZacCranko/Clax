import os, sys, time
from typing import Any, Callable
from absl import logging
from tensorflow_datasets.core import dataset_builder
import jax, flax, optax, ml_collections, functools

from flax.training import common_utils, train_state
from jax import numpy as jnp, random, lax, tree_util
from flax import jax_utils, struct

import tensorflow as tf
import tensorflow_datasets as tfds

from flax import struct

class EncoderState(struct.PyTreeNode):
  apply_fn: Callable = struct.field(pytree_node=False)
  params: flax.core.FrozenDict[str, Any]

  @classmethod
  def create(cls, apply_fn, params):
    return cls(apply_fn=apply_fn, params=params)

def create_train_state(rng, clf_config, num_features, num_classes):
  # initialise model
  clf = flax.linen.Dense(num_classes)
  params = clf.init(rng, jnp.ones((128, num_features)))

  tx = optax.lars(
        learning_rate=clf_config.learning_rate,
        momentum=clf_config.momentum,
        nesterov=True,
    )
  return train_state.TrainState.create(apply_fn=clf.apply, params = params, tx=tx)

@jax.jit
def compute_metrics(*, logits, labels):
  metrics = {
    'cross_entropy_loss' : optax.softmax_cross_entropy(logits = logits, labels = labels).mean(),
    'accuracy' :  jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
  }
  return metrics

@jax.jit
def l2_reg(params, coeff: jnp.float32):
  squared_params = tree_util.tree_map(lambda x: x*x, params)
  sum = tree_util.tree_reduce(lambda x, y: jnp.sum(x) + jnp.sum(y), squared_params)
  return coeff * sum

@functools.partial(jax.pmap, axis_name = "batch")
def train_step(encod_state, clf_state, images, labels, coeff):
  def loss_fn(params):
    encodings = encod_state.apply_fn(encod_state.params, images, train = False)
    
    logits = clf_state.apply_fn(params, encodings)

    metrics = compute_metrics(logits = logits, labels = labels)
    loss = metrics['cross_entropy_loss'] + l2_reg(params, coeff = coeff)

    return loss, metrics

  grad_fn = jax.value_and_grad(loss_fn, has_aux = True)

  (_, metrics), grads = grad_fn(clf_state.params)

  clf_state = clf_state.apply_gradients(grads = lax.pmean(grads, axis_name = "batch"))
  return clf_state, metrics

def evaluate(rng: random.PRNGKey, 
             clf_config: ml_collections.ConfigDict, 
             state: train_state.TrainState, 
             assembly: flax.linen.Module, 
             image_shape: int, 
             num_classes: int, 
             steps_per_epoch: int, dataset):
  num_steps = clf_config.num_epochs * steps_per_epoch
  
  params = flax.core.frozen_dict.freeze({"params" : state.params["backbone"], "batch_stats" : state.batch_stats["backbone"] })
  encod_state = EncoderState.create(apply_fn = assembly.backbone.apply, params = params)

  _, num_features = encod_state.apply_fn(encod_state.params, jnp.ones((128, ) + image_shape), train = False).shape
  
  clf_state = create_train_state(rng, clf_config, num_features, num_classes)

  clf_state = jax_utils.replicate(clf_state)
  encod_state = jax_utils.replicate(encod_state)

  l2coeff = jnp.repeat(clf_config.l2coeff, jax.local_device_count())

  logging.info("Evaluating linear accuracy")

  train_metrics = []
  for step, batch in zip(range(num_steps), dataset):
    clf_state, metrics = train_step(encod_state, clf_state, *batch, l2coeff)

    if (step + 1) % steps_per_epoch == 0:
      epoch = step // steps_per_epoch
    
      train_metrics.append(metrics)
      get_train_metrics = common_utils.get_metrics(train_metrics)
      summary = {
              f'train_{k}': v
              for k, v in jax.tree_map(lambda x: x.mean(), get_train_metrics).items()
          }

  
  train_metrics = jax.tree_map(lambda x: x.mean(), train_metrics)
  max_accuracy_epoch = jnp.argmax(jnp.array([x['accuracy'] for x in train_metrics]))
  max_accuracy = train_metrics[max_accuracy_epoch]['accuracy']
  
  logging.info("Best accuracy: {0:.2%}, in epoch {1}".format(max_accuracy, max_accuracy_epoch + 1))

  metrics = {
    "max_accuracy" : max_accuracy,
    "train_metrics" : train_metrics
  }
  
  return metrics

#%%
# import defaults
# import train
# config = defaults.get_config()

# clf_config = config.clf_config 
# clf_config.cache_dataset = True
# clf_config.l2coeff = 0.001
# clf_config.num_epochs = 5
# clf_config.batch_size = 1024

# rng = random.PRNGKey(0)
# image_shape = (32,32,3)
# num_classes = 10


# %%

# assembly = train.create_model(config)
# state = train.create_train_state(rng, config, assembly = assembly, image_shape = (32,32,3), learning_rate_fn = 0.1)


# #%%
# logging.set_verbosity(logging.INFO)
# dataset_builder = tfds.builder(config.dataset)
# num_examples = dataset_builder.info.splits['train'].num_examples
# image_shape  = dataset_builder.info.features['image'].shape
# steps_per_epoch    = num_examples // config.batch_size 
# base_learning_rate = config.learning_rate * config.batch_size / 256.

# train_dataset = train.create_input_iter(clf_config, dataset_builder, is_training = False)

# metrics = evaluate(rng, clf_config, state, assembly, image_shape, num_classes, steps_per_epoch, train_dataset)
