#%%
import os, sys, time
from typing import Any, Callable
from absl import logging
from tensorflow_datasets.core import dataset_builder
import jax, flax, optax, ml_collections, functools

import torchvision.transforms as transforms
from flax.training import common_utils, train_state
from jax import numpy as jnp, random, lax, tree_util
from flax import jax_utils

#%%

class LinEvalTrainState(train_state.TrainState):
  encoder_fn: Callable


def create_train_state(rng, num_classes, learning_rate, momentum, dtype):

  # initialise model
  clf = flax.linen.Dense(num_classes, dtype = dtype)
  params = clf.init(rng, jnp.ones((128, num_classes)))

  # initialise calssifier
  tx = optax.sgd(learning_rate, momentum)

  return train_state.TrainState.create(apply_fn=clf.apply, params = params, tx=tx)

@jax.jit
def compute_metrics(*, logits, labels):
  metrics = {
    'cross_entropy_loss' : optax.softmax_cross_entropy(logits = logits, labels = labels).mean(),
    'accuracy' :  jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
  }
  return metrics

@jax.jit
def l2_regularise(params, coeff):
  squared_params = tree_util.tree_map(lambda x: x*x, params)
  sum = tree_util.tree_reduce(lambda x, y: jnp.sum(x) + jnp.sum(y), squared_params)
  return coeff * sum

@functools.partial(jax.pmap, axis_name = "batch")
def train_step(state, images, labels):
  def loss_fn(parms):
    logits = state.apply_fn(params = state.params, images)
    metrics = compute_metrics(logits = logits, labels = labels)
    loss = metrics['cross_entropy_loss']

    loss = loss + l2_regularise(state.params, coeff = 0.1)

    return loss, metrics

  grad_fn = jax.value_and_grad(loss_fn, has_aux = True)

  (_, metrics), grads = grad_fn(state.params)

  state = state.apply_gradients(grads = lax.pmean(grads, axis_name = "batch"))
  return state, metrics

def train_and_evaluate(state, train_ds, batch_size, num_steps):
  for step, batch in zip(range(num_steps), train_ds):
    state, metrics = train_step(state, *batch)

