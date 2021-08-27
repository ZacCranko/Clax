from typing import Any, Callable
from absl import logging
import jax, flax, optax, ml_collections, functools

from flax.training import common_utils, train_state
from jax import numpy as jnp, random, lax
from flax import jax_utils, struct, struct

import data, objective

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
                       num_classes: int, clf_state = None):
  
  params = flax.core.frozen_dict.freeze({"params" : state.params["backbone"], "batch_stats" : state.batch_stats["backbone"]})
  encod_state = EncoderState.create(apply_fn = assembly.backbone.apply, params = params)
  _, num_features = encod_state.apply(jnp.ones((128,) + image_shape), train = False).shape
  
  # initialise model
  if clf_state is not None and clf_config.warm_start:
    clf = flax.linen.Dense(num_classes)
    params = clf_state.params
  else:
    clf, params = initialize(rng, num_classes, num_features)
  
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
                  train_iter: data.TrainIterator, clf_state = None):

  # build the encoder state and deduce the number of features
  encod_state, clf_state = create_train_state(rng, clf_config, state, assembly, 
                                              train_iter.image_shape, train_iter.num_classes,
                                              clf_state = clf_state)

  # replicate state parameters
  encod_state = jax_utils.replicate(encod_state)
  clf_state   = jax_utils.replicate(clf_state)
  l2coeff     = jax_utils.replicate(clf_config.l2coeff)

  
  epoch_metrics = []
  metrics_by_batch = []
  for batch in train_iter(info = f"Linear evaluation"):
    clf_state, metrics = train_step(encod_state, clf_state, l2coeff, *batch)
    metrics_by_batch.append(jax.tree_map(jnp.mean, metrics))

    if train_iter.is_epoch_end():
      epoch_metrics.append(jax.tree_map(jnp.mean, metrics_by_batch))
      metrics_by_batch = []
  
  # max_accuracy_epoch = jnp.argmax(jnp.array([x['accuracy'] for x in epoch_metrics]))
  # max_accuracy = epoch_metrics[max_accuracy_epoch]['accuracy']
  # logging.info(f"Linear evaluation accuracy: {metrics['accuracy']:.2%}, in epoch {max_accuracy_epoch + 1} of {clf_config.num_epochs}")
  
  metrics = epoch_metrics[-1][-1]
  
  logging.info(f"Linear evaluation accuracy: {metrics['accuracy']:.2%}")
  print(metrics)
  return metrics, jax_utils.unreplicate(clf_state)

def grid_search_evaluate(key: random.PRNGKey, 
                         clf_config: ml_collections.ConfigDict,
                         state: train_state.TrainState,
                         assembly: flax.linen.Module,
                         train_iter: data.TrainIterator,
                         *, start: float, stop: float, num: int):
  # copy the config dict so that we can overwrite the l2coeff
  _clf_config = ml_collections.ConfigDict(clf_config.to_dict())
  l2coeff_grid_search = jnp.linspace(start, stop, num = num)

  grid_search_metrics = {}
  for coeff in l2coeff_grid_search(info="L2 coefficient grid search"):
    key, subkey = random.split(key)
    _clf_config.l2coeff = coeff
    grid_search_metrics[f"l2coeff_{coeff}"] = fast_evaluate(subkey, _clf_config, state, assembly, train_iter)

  return grid_search_metrics
