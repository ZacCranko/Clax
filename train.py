# %% 1: Imports
from absl import logging

from typing import Any, Callable, Dict
from flax.training.train_state import TrainState
from data.iterator import TrainIterator
from ml_collections import ConfigDict
from jax.random import PRNGKey

import jax, functools

from jax import numpy as jnp, lax

from flax import jax_utils
from flax.training import checkpoints as chkp
from flax.metrics import tensorboard

import objective as obj, data, init
# import linear_evaluation as eval
# import linear_evaluation_lbfgs as lbfgs_eval
from evaluation import linear_accuracy
import wandb


def get_key():
  global key
  subkey, key = jax.random.split(key)
  return subkey


class SummaryWriter(tensorboard.SummaryWriter):

  def log(self, summary, step: int):
    for tag, value in summary.items():
      self.scalar(tag=tag, value=value, step=step)


cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def train_step(
    state: TrainState,
    device_id: int,
    images,
    labels,
    learning_rate_fn: Callable,
    temp: float,
    unif_coeff: float,
    axis_name: str,
    is_supervised: bool = False,
):
  """Perform a single training step."""

  def loss_fn(params):
    """loss function used for training."""
    projections, new_model_state = state.apply_fn(
        {
            "params": params,
            "batch_stats": state.batch_stats
        },
        images,
        mutable=["batch_stats"],
    )

    metrics = dict()

    ntxent, (align, unif) = obj.ntxent(
        device_id,
        projections,
        temp=temp,
        unif_coeff=unif_coeff,
        axis_name=axis_name,
    )
    loss = ntxent

    if is_supervised:
      clf_metrics = obj.classification_metrics(logits=projections,
                                               labels=labels)
      loss = clf_metrics["cross_entropy"]

      for k, v in clf_metrics.items():
        metrics[k] = v

    metrics["align"] = align
    metrics["unif"] = unif
    metrics["ntxent"] = ntxent

    return loss, (new_model_state, metrics)

  step = state.step

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (new_model_state, metrics)), grads = grad_fn(state.params)

  grads = lax.pmean(grads, axis_name=axis_name)

  new_state = state.apply_gradients(grads=grads,
                                    batch_stats=new_model_state["batch_stats"])
  metrics["learning_rate"] = learning_rate_fn(step)

  return new_state, lax.pmean(metrics, axis_name=axis_name)


def save_checkpoint(config: ConfigDict, workdir: str, state: TrainState):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax_utils.unreplicate(state))
    step = int(state.step)
    chkp.save_checkpoint(workdir,
                         state,
                         step,
                         keep=config.num_checkpoints_to_keep)


def train_and_evaluate(config: ConfigDict, workdir: str) -> TrainState:
  global key
  key = PRNGKey(config.seed)

  summary_writer = SummaryWriter(workdir)
  summary_writer.hparams(config.to_dict())

  train_iter = data.config_create_input_iter(config, num_transforms=2)

  learning_rate_fn = init.create_learning_rate_fn(config,
                                                  train_iter.steps_per_epoch)

  # initialise model
  assembly = init.create_assembly(config, axis_name="batch")
  state = init.create_train_state(get_key(), config, assembly,
                                  train_iter.image_shape, learning_rate_fn,
                                  workdir)

  linear_train_iter = data.config_create_input_iter(
      config.clf_config,
      dataset=config.dataset,
      is_pretrain=False,
  )
  linear_valid_iter = data.config_create_input_iter(
      config.clf_config,
      dataset=config.dataset,
      split="test",
      is_pretrain=False,
  )

  # replicate parameters to xla devices
  state_repl = jax_utils.replicate(state)
  device_ids = jnp.arange(jax.local_device_count())

  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          temp=config.ntxent_temp,
          unif_coeff=config.ntxent_unif_coeff,
          learning_rate_fn=learning_rate_fn,
          axis_name="batch",
          is_supervised=config.is_supervised,
      ),
      axis_name="batch",
  )

  logging.info("Starting training")
  for batch in train_iter:
    state_repl, metrics = p_train_step(state_repl, device_ids, *batch)
    wandb.log({"train": jax.tree_map(lambda x: x.mean(), metrics)},
              commit=False)

    if train_iter.is_start():
      logging.info("Model is compiled")

    # linear evaluation
    if train_iter.is_freq(step_freq=config.linear_eval_freq):
      accuracy = linear_accuracy(
          state_repl,
          linear_train_iter,
          linear_valid_iter,
      )

      wandb.log({"test": {"linear_accuracy": accuracy}}, commit=False)

    if train_iter.is_freq(step_freq=config.checkpoint_freq):
      save_checkpoint(config, workdir, state_repl)

    wandb.log(dict(), step=train_iter.global_step)

  # Wait until computations are done before exiting
  jax.random.normal(PRNGKey(0), ()).block_until_ready()

  return jax_utils.unreplicate(state_repl)
