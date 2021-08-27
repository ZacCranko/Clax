# %% 1: Imports
import os, sys, time
from typing import Any, Callable, Dict
from absl import logging 

import jax, ml_collections, functools, optax
import tensorflow_datasets as tfds

from jax import numpy as jnp, lax
from jax.random import PRNGKey
from flax import jax_utils

from flax.training import common_utils, checkpoints as chkp
from flax.metrics import tensorboard
import linear_evaluation as eval


import objective as obj, data, init

def get_key():
  global key
  subkey, key = jax.random.split(key)
  return subkey

class SummaryWriter(tensorboard.SummaryWriter):
  def log(self, summary, step: int):
    for tag, value in summary.items():
      self.scalar(tag = tag, value = value, step = step)

cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")

def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

def train_step(state: init.CLTrainState,
               device_id: int, 
               images, labels,
               learning_rate_fn: Callable,
               temp: float, axis_name: str):
    """Perform a single training step."""

    def loss_fn(params):
        """loss function used for training."""
        projections, new_model_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            images, mutable=["batch_stats"],
        )

        loss, (align, unif) = obj.ntxent(device_id, projections, temp = temp, axis_name = axis_name)
        metrics = {}
        
        # clf_metrics = obj.classification_metrics(logits = projections, labels = labels)

        metrics['align'] = align
        metrics['unif'] = unif

        # loss = metrics['cross_entropy']

        return loss, (new_model_state, metrics)

    step = state.step

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_model_state, metrics)), grads = grad_fn(state.params)

    grads = lax.pmean(grads, axis_name = axis_name)

    new_state = state.apply_gradients(
        grads=grads, batch_stats = new_model_state["batch_stats"]
    )
    metrics['learning_rate'] = learning_rate_fn(step)

    return new_state, lax.pmean(metrics, axis_name = axis_name)

def save_checkpoint(workdir, state):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax_utils.unreplicate(state))
    step  = int(state.step)
    chkp.save_checkpoint(workdir, state, step, keep=5)

def train_and_evaluate(_key: PRNGKey, config: ml_collections.ConfigDict, workdir: str) -> init.CLTrainState:
  global key 
  key = _key 

  summary_writer = SummaryWriter(workdir)
  summary_writer.hparams(config.to_dict())

  train_iter = data.create_input_iter(config, is_training = True)
  
  learning_rate_fn = init.create_learning_rate_fn(config, train_iter.steps_per_epoch)

  # initialise model
  assembly = init.create_assembly(config, axis_name = "batch")
  state = init.create_train_state(
      get_key(), config, assembly, train_iter.image_shape, learning_rate_fn
  )

  linear_train_iter = data.create_input_iter(config.clf_config, is_training = False, dataset = config.dataset)
  linear_test_iter  = data.create_input_iter(config.clf_config, is_training = False, dataset = config.dataset, split = 'test')

  # replicate parameters to xla devices
  rep_state = jax_utils.replicate(state)
  device_ids = jnp.arange(jax.local_device_count())

  p_train_step = jax.pmap(
      functools.partial(train_step, 
                        temp = config.ntxent_temp, unif_coeff = config.ntxent_unif_coeff,
                        learning_rate_fn = learning_rate_fn, 
                        axis_name = "batch"), 
      axis_name = "batch")

  logging.info("Starting training")
  for batch in train_iter:
    rep_state, metrics = p_train_step(rep_state, device_ids, *batch)
    summary = {
            f'train_{k}': v
            for k, v in jax.tree_map(lambda x: x.mean(), metrics).items()
        }
    summary = train_iter.append_metrics(summary)
    summary_writer.log(summary, train_iter.global_step)

    # linear evaluation
    if train_iter.is_freq(step_freq = config.linear_eval_step_freq):
      metrics = eval.linear_accuracy(get_key(), config, jax_utils.unreplicate(rep_state), assembly, linear_train_iter, linear_test_iter)
      summary_writer.scalar("train_linear_accuracy", metrics["accuracy"], train_iter.global_step)

    # checkpoint model
    if train_iter.is_freq(step_freq = config.checkpoint_step_freq):
      save_checkpoint(workdir, rep_state)

  # Wait until computations are done before exiting
  jax.random.normal(PRNGKey(0), ()).block_until_ready()

  return rep_state

