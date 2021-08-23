import os, sys, time
from typing import Any, Callable
from absl import logging
import jax, flax, optax, ml_collections, functools

from flax.training import common_utils, train_state
from jax import numpy as jnp, random, lax, tree_util
from flax import jax_utils, struct, struct

import wandb
import train, data, objective, init

class EncoderState(struct.PyTreeNode):
  apply_fn: Callable = struct.field(pytree_node=False)
  params: flax.core.FrozenDict[str, Any]

  def apply(self, x, train: bool = True):
    return self.apply_fn(self.params, x, train = train)

  @classmethod
  def create(cls, **kwargs):
    return cls(**kwargs)

def initialize(rng: random.PRNGKey, num_classes: int, num_features: int):
  clf = flax.linen.Dense(num_classes)
  
  @jax.jit
  def init(*args):
      return clf.init(*args)

  params = init(rng, jnp.ones((128, num_features)))
  return clf, params

def create_train_state(rng: random.PRNGKey, 
                       clf_config: ml_collections.ConfigDict, 
                       state, assembly, image_shape, 
                       num_classes: int,
                       learning_rate_fn):
  
  params = flax.core.frozen_dict.freeze({"params" : state.params["backbone"], "batch_stats" : state.batch_stats["backbone"]})
  encod_state = EncoderState.create(apply_fn = assembly.backbone.apply, params = params)
  _, num_features = encod_state.apply(jnp.ones((128,) + image_shape), train = False).shape
  
  # initialise model
  clf, params = initialize(rng, num_classes, num_features)

  # tx = optax.lars(
  #       learning_rate=learning_rate_fn,
  #       momentum=clf_config.momentum,
  #       nesterov=True,
  #   )
  tx = optax.adam(learning_rate=clf_config.learning_rate)
  clf_state = train_state.TrainState.create(apply_fn=clf.apply, params = params, tx=tx)
  return encod_state, clf_state

@functools.partial(jax.pmap, axis_name = "batch")
def train_step(encod_state: EncoderState, 
               clf_state: train_state.TrainState, 
               coeff: float,
               images, labels):

  def loss_fn(params):
    encodings = encod_state.apply(images, train = False)
    logits = clf_state.apply_fn(params, encodings)
    metrics = objective.classification_metrics(logits = logits, labels = labels)
    loss = metrics['cross_entropy'] + objective.l2_reg(params, coeff = coeff)

    return loss, metrics

  grad_fn = jax.value_and_grad(loss_fn, has_aux = True)

  (_, metrics), grads = grad_fn(clf_state.params)

  clf_state = clf_state.apply_gradients(grads = lax.pmean(grads, axis_name = "batch"))
  return clf_state, metrics

def fast_evaluate(rng: random.PRNGKey, 
                  clf_config: ml_collections.ConfigDict, 
                  state: train_state.TrainState, 
                  assembly: flax.linen.Module, 
                  train_iter: data.DatasetIterator):

  # build the encoder state and deduce the number of features
  learning_rate_fn = init.create_learning_rate_fn(clf_config, train_iter.steps_per_epoch)
  encod_state, clf_state = create_train_state(rng, clf_config, state, assembly, 
                                              train_iter.image_shape, train_iter.num_classes, learning_rate_fn)

  logging.info("Evaluating linear accuracy")
  
  # replicate state parameters
  encod_state = jax_utils.replicate(encod_state)
  clf_state   = jax_utils.replicate(clf_state)
  l2coeff     = jax_utils.replicate(clf_config.l2coeff)

  train_metrics = []
  for batch in train_iter(info = "Linear evaluation loop"):
    clf_state, metrics = train_step(encod_state, clf_state, l2coeff, *batch)

    if train_iter.is_epoch_start():
      train_metrics.append(metrics)
      get_train_metrics = common_utils.get_metrics(train_metrics)
      summary = {
              f'linear_eval_train_{k}': v
              for k, v in jax.tree_map(lambda x: x.mean(), get_train_metrics).items()
          }
      summary = train_iter.append_metrics(summary, prefix = "linear_eval.train_")
      wandb.log(summary)
  
  train_metrics = jax.tree_map(lambda x: x.mean(), train_metrics)
  max_accuracy_epoch = jnp.argmax(jnp.array([x['accuracy'] for x in train_metrics]))
  max_accuracy = train_metrics[max_accuracy_epoch]['accuracy']
  logging.info(f"Best accuracy: {max_accuracy:.2%}, in epoch {max_accuracy_epoch + 1} of {clf_config.num_epochs}")
  
  metrics = {
    "max_accuracy" : max_accuracy,
    "train_metrics" : train_metrics
  }
  
  return metrics

def grid_search_evaluate(key: random.PRNGKey, 
                         clf_config: ml_collections.ConfigDict,
                         state: train_state.TrainState,
                         assembly: flax.linen.Module,
                         train_iter: data.DatasetIterator,
                         *, start: float, stop: float, num: int):
  # copy the config dict so that we can overwrite the l2coeff
  #                          
  _clf_config = ml_collections.ConfigDict(clf_config.to_dict())
  l2coeff_grid_search = jnp.linspace(start, stop, num = num)

  grid_search_metrics = {}
  logging.info("Starting L2 coefficient grid search")
  for coeff in l2coeff_grid_search:
    key, subkey = random.split(key)
    _clf_config.l2coeff = coeff
    grid_search_metrics[f"l2coeff_{coeff}"] = fast_evaluate(subkey, _clf_config, state, assembly, train_iter)

  return grid_search_metrics
