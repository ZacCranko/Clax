# %% 1: Imports
import os, sys, time
from typing import Any, Callable

import jax, ml_collections, functools

from flax.training import common_utils
from jax import numpy as jnp, random, lax
from flax import jax_utils

import tensorflow_datasets as tfds

import objective, data, linear_evaluation, serialization, init
import wandb

def prepare_batch(batch):
    aug_images, labels = batch

    local_device_count = jax.local_device_count()
    ch = 3
    
    # images is b × h × w × (ch * num_augs)
    aug_images = aug_images._numpy()
    num_augs = aug_images.shape[3] // ch
    # aug_list is (b * num_augs) × h × w × ch
    images = jnp.concatenate(jnp.split(aug_images, num_augs, axis = -1))

    labels = labels._numpy()
    labels = jnp.concatenate([labels for _ in range(num_augs)], axis = 0)

    # xla split is xla × (b * num_augs)/xla × h × w × ch
    xla_split_images = images.reshape((local_device_count, -1) + images.shape[1:])
    xla_split_labels = labels.reshape((local_device_count, -1) + labels.shape[1:])
    return xla_split_images, xla_split_labels

def create_input_iter(config: ml_collections.ConfigDict, is_training: bool, split: str = 'train', dataset = None):
  dataset_builder = tfds.builder(dataset if dataset is not None else config.dataset)
  ds = data.get_dataset(dataset_builder, batch_size = config.batch_size, is_training = is_training, cache_dataset = config.cache_dataset)
  ds_map = map(prepare_batch, ds)

  dataset_iter = data.DatasetIterator(dataset_builder = dataset_builder, dataset_iter = ds_map, batch_size = config.batch_size, 
                                      split = split, num_steps = config.num_steps, num_epochs = config.num_epochs, start_step = config.start_step)

  return dataset_iter

cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")

def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

def train_step(state: init.TrainState,
               device_id: int, 
               images, labels,
               learning_rate_fn: Callable,
               temp: float,
               axis_name: str = "batch"):
    """Perform a single training step."""

    def loss_fn(params):
        """loss function used for training."""
        projections, new_model_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            images, mutable=["batch_stats"],
        )
        
        loss, (align, unif) = objective.ntxent(device_id, projections, temp = temp, axis_name = axis_name)
        
        metrics = {
          "loss" : loss,
          "align" : align,
          "unif" : unif,
        }

        return loss, (new_model_state, metrics)

    step = state.step

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_model_state, metrics)), grads = grad_fn(state.params)

    grads = lax.pmean(grads, axis_name = axis_name)

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )
    
    metrics['learning_rate'] = learning_rate_fn(step)
    return new_state, lax.pmean(metrics, axis_name = axis_name)

def train_and_evaluate(config: ml_collections.ConfigDict) -> init.TrainState:
  key = random.PRNGKey(0)


  train_iter = create_input_iter(config, is_training = True)
  linear_train_iter = create_input_iter(config.clf_config, is_training = False, dataset = config.dataset)

  learning_rate_fn = init.create_learning_rate_fn(config, train_iter.steps_per_epoch)

  # initialise model
  subkey, key = random.split(key)

  assembly = init.create_assembly(config, axis_name = "batch")
  state = init.create_train_state(
      subkey, config, assembly, train_iter.image_shape, learning_rate_fn
  )

  # replicate parameters to xla devices
  state = jax_utils.replicate(state)
  device_ids = jnp.arange(jax.local_device_count())

  p_train_step = jax.pmap(
      functools.partial(train_step, temp = config.ntxent_temp, learning_rate_fn = learning_rate_fn),
      axis_name="batch")


  def linear_eval(key, epoch):
    linear_metrics = linear_evaluation.fast_evaluate(key, config.clf_config, jax_utils.unreplicate(state), 
                                                assembly, linear_train_iter)
    wandb.log({
        "epoch" : epoch,
        "train_linear_accuracy" : linear_metrics["max_accuracy"]
      })
  
  for batch in train_iter(info = "Main train loop"):
    train_metrics = []
    train_metrics_last_t = time.time()
    
    state, metrics = p_train_step(state, device_ids, *batch)

    if train_iter.is_freq(step_freq = config.linear_eval_step_freq):
      subkey, key = random.split(key)
      linear_eval(subkey, train_iter.get_epoch())

    if train_iter.is_freq(step_freq = config.save_projector_step_freq):
      serialization.save_projector(config, jax_utils.unreplicate(state), step = train_iter.global_step)

    if train_iter.is_epoch_start():
      # sync batch statistics across replicas
      state = sync_batch_stats(state)
    
    train_metrics.append(metrics)
    get_train_metrics = common_utils.get_metrics(train_metrics)
    summary = {
            f'train_{k}': v
            for k, v in jax.tree_map(lambda x: x.mean(), get_train_metrics).items()
        }
    summary = train_iter.append_metrics(summary)
    
    wandb.log(summary)
    train_metrics = []
    
  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return state

