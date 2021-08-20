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

import train

class EncoderState(struct.PyTreeNode):
  apply_fn: Callable = struct.field(pytree_node=False)
  params: flax.core.FrozenDict[str, Any]

  @classmethod
  def create(cls, apply_fn, params):
    return cls(apply_fn=apply_fn, params=params)

def initialize(rng: random.PRNGKey, num_classes: int, num_features: int):
  clf = flax.linen.Dense(num_classes)
  params = clf.init(rng, jnp.ones((128, num_features)))
  return clf, params

def create_train_state(rng: random.PRNGKey, 
                       clf_config: ml_collections.ConfigDict, 
                       state, image_shape, 
                       num_classes: int,
                       learning_rate_fn):
  
  params = flax.core.frozen_dict.freeze({"params" : state.params["backbone"], "batch_stats" : state.batch_stats["backbone"] })
  encod_state = EncoderState.create(apply_fn = assembly.backbone.apply, params = params)
  _, num_features = encod_state.apply_fn(encod_state.params, jnp.ones((128, ) + image_shape), train = False).shape
  
  # initialise model
  clf, params = initialize(rng, num_classes, num_features)

  tx = optax.lars(
        learning_rate=learning_rate_fn,
        momentum=clf_config.momentum,
        nesterov=True,
    )
  clf_state
  return encod_state, clf_state

@jax.jit
def compute_metrics(*, logits, labels):
  metrics = {
    'cross_entropy' : optax.softmax_cross_entropy(logits = logits, labels = labels).mean(),
    'accuracy' :  jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
  }
  return metrics

@jax.jit
def l2_reg(params, coeff: float):
  squared_params = tree_util.tree_map(lambda x: x*x, params)
  sum = tree_util.tree_reduce(lambda x, y: jnp.sum(x) + jnp.sum(y), squared_params)
  return coeff * sum

@functools.partial(jax.pmap, axis_name = "batch")
def train_step(encod_state: EncoderState, 
               clf_state: train_state.TrainState, 
               coeff: float,
               images, labels):

  def loss_fn(params):
    encodings = encod_state.apply_fn(encod_state.params, images, train = False)
    logits = clf_state.apply_fn(params, encodings)
    metrics = compute_metrics(logits = logits, labels = labels)
    loss = metrics['cross_entropy'] + l2_reg(params, coeff = coeff)

    return loss, metrics

  grad_fn = jax.value_and_grad(loss_fn, has_aux = True)

  (_, metrics), grads = grad_fn(clf_state.params)

  clf_state = clf_state.apply_gradients(grads = lax.pmean(grads, axis_name = "batch"))
  return clf_state, metrics


def create_states(state, image_shape):
  params = flax.core.frozen_dict.freeze({"params" : state.params["backbone"], "batch_stats" : state.batch_stats["backbone"] })
  encod_state = EncoderState.create(apply_fn = assembly.backbone.apply, params = params)

  _, num_features = encod_state.apply_fn(encod_state.params, jnp.ones((128, ) + image_shape), train = False).shape

  learning_rate_fn = train.create_learning_rate_fn(clf_config, base_learning_rate, steps_per_epoch)
  clf_state = create_train_state(rng, clf_config, num_features, num_classes, learning_rate_fn)

  return encod_state, clf_state

def evaluate(rng: random.PRNGKey, 
             clf_config, 
             state: train_state.TrainState, 
             assembly: flax.linen.Module, 
             dataset_builder, train_dataset):

  num_examples = dataset_builder.info.splits['train'].num_examples
  image_shape  = dataset_builder.info.features['image'].shape
  num_classes  = dataset_builder.info.features['label'].num_classes
  steps_per_epoch    = num_examples // clf_config.batch_size 
  base_learning_rate = clf_config.learning_rate * clf_config.batch_size / 256.
  num_steps = clf_config.num_epochs * steps_per_epoch
  
  # build the encoder state and deduce the number of features
  learning_rate_fn = train.create_learning_rate_fn(clf_config, base_learning_rate, steps_per_epoch)
  encod_state, clf_state = create_train_state(rng, clf_config, state, image_shape, num_classes, learning_rate_fn)

  logging.info("Evaluating linear accuracy")
  
  # replicate state parameters
  encod_state = jax_utils.replicate(encod_state)
  clf_state   = jax_utils.replicate(clf_state)
  l2coeff     = jax_utils.replicate(clf_config.l2coeff)

  train_metrics = []
  for step, batch in zip(range(num_steps), train_dataset):
    clf_state, metrics = train_step(encod_state, clf_state, l2coeff, *batch)

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
  logging.info(f"Best accuracy: {max_accuracy:.2%}, in epoch {max_accuracy_epoch + 1} of {clf_config.num_epochs}")
  
  metrics = {
    "max_accuracy" : max_accuracy,
    "train_metrics" : train_metrics
  }
  
  return metrics