# %% 1: Imports
import os, sys, time
from typing import Any
from absl import logging
from tensorflow_datasets.core import dataset_builder
import jax, flax, optax, ml_collections, functools

import torchvision.transforms as transforms
from flax.training import common_utils, train_state
from jax import numpy as jnp, random, lax
from flax import jax_utils

import tensorflow as tf
import tensorflow_datasets as tfds

import defaults, objective, models, data
import wandb

# %% 2: Define model
def create_model(config: ml_collections.ConfigDict):
    is_tpu_platform = jax.local_devices()[0].platform == "tpu"
    if config.half_precision:
        model_dtype = jnp.bfloat16 if is_tpu_platform else jnp.float16
    else:
        model_dtype = jnp.float32

    assembly = models.projector.Assembly(stem = getattr(models.stem, config.stem),
                                         backbone = getattr(models.resnet, config.model), 
                                         projector = getattr(models.projector, config.projector), 
                                         dtype = model_dtype)
    return assembly

# generates model variable placeholders
def initialized(rng, image_shape, model):
    input_shape = (128, ) + image_shape

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init(rng, jnp.ones(input_shape, model.dtype))
    return variables["params"], variables["batch_stats"]

# %% 5: Loading data
def prepare_batch(batch):
    aug_images, labels = batch

    local_device_count = jax.local_device_count()
    ch = 3
    
    # images is b × h × w × (ch * num_augs)
    aug_images = aug_images._numpy()
    num_augs = aug_images.shape[3] // ch
    aug_list = jnp.split(aug_images, num_augs, axis = -1)
    
    # aug_list is (b * num_augs) × h × w × ch
    images = jnp.concatenate(aug_list)

    labels = labels._numpy()
    labels = jnp.repeat(labels, images.shape[0] // labels.shape[0], axis = 0)

    # xla split is xla × (b * num_augs)/xla × h × w × ch
    xla_split_images = images.reshape((local_device_count, -1) + images.shape[1:])
    xla_split_labels = labels.reshape((local_device_count, -1) + labels.shape[1:])
    return xla_split_images, xla_split_labels

def create_input_iter(config, dataset_builder, is_training):
  ds = data.get_dataset(dataset_builder, batch_size = config.batch_size, is_training = is_training, cache_dataset = config.cache_dataset)
  it = map(prepare_batch, ds)
  return it


# %% 6: Create train state
class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: flax.optim.DynamicScale

def create_learning_rate_fn(
    config: ml_collections.ConfigDict, base_learning_rate: float, steps_per_epoch: int
):
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch,
    )
    
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)

    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )

    learning_rate_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch],
    )

    return learning_rate_fn

def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_shape, learning_rate_fn
) -> TrainState:
    """Create initial training state."""
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == "gpu":
        dynamic_scale = flax.optim.DynamicScale()
    else:
        dynamic_scale = None

    params, batch_stats = initialized(rng, image_shape, model)
    tx = optax.lars(
        learning_rate=learning_rate_fn,
        momentum=config.momentum,
        nesterov=True,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=tx,
        dynamic_scale=dynamic_scale,
    )
    return state

# %% 7: Training step
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")

def compute_metrics(logits, one_hot_labels):
  cross_entropy = jnp.mean(optax.softmax_cross_entropy(logits, one_hot_labels))
  accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(one_hot_labels, -1))
  metrics = {
      'cross_entropy': cross_entropy,
      'accuracy': accuracy,
  }
  return metrics

def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

# axis_name = "batch"
def train_step(state, device_id, images, labels, learning_rate_fn):
    """Perform a single training step."""

    def loss_fn(params):
        """loss function used for training."""
        logits, new_model_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            images, mutable=["batch_stats"],
        )
        # loss, (align, unif) = objective.ntxent(device_id, logits, temp = 0.5)
        metrics = compute_metrics(logits, labels)

        loss = metrics['cross_entropy']

        return loss, (new_model_state, metrics)

    step = state.step

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_model_state, metrics)), grads = grad_fn(state.params)

    grads = lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )
    
    metrics['learning_rate'] = learning_rate_fn(step)
    return new_state, lax.pmean(metrics, axis_name ="batch")

#%%


def train_and_evaluate(config: ml_collections.ConfigDict) -> TrainState:
  rng = random.PRNGKey(0)

  dataset_builder = tfds.builder(config.dataset)
  
  num_examples = dataset_builder.info.splits['train'].num_examples
  image_shape  = dataset_builder.info.features['image'].shape
  steps_per_epoch    = num_examples // config.batch_size 
  base_learning_rate = config.learning_rate * config.batch_size / 256.

  train_dataset = create_input_iter(config, dataset_builder, is_training = True)

  model = create_model(config)

  learning_rate_fn = create_learning_rate_fn(
      config, base_learning_rate, steps_per_epoch)

  state = create_train_state(
      random.PRNGKey(0), config, model, image_shape, learning_rate_fn
  )

  step_offset = int(state.step)

  state = jax_utils.replicate(state)

  device_ids = jnp.arange(jax.local_device_count())
  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn = learning_rate_fn),
      axis_name="batch")
  
  num_steps = int(steps_per_epoch * config.num_epochs)
  
  logging.info('Compiling model')
  for step, batch in zip(range(step_offset, num_steps), train_dataset):
    train_metrics = []
    train_metrics_last_t = time.time()
    
    state, metrics = p_train_step(state, device_ids, *batch)
    
    if (step + 1) % steps_per_epoch == 0:
      epoch = step // steps_per_epoch

      # sync batch statistics across replicas
      state = sync_batch_stats(state)

    if step == step_offset:
      logging.info('Training')
    
    train_metrics.append(metrics)
    get_train_metrics = common_utils.get_metrics(train_metrics)
    summary = {
            f'train_{k}': v
            for k, v in jax.tree_map(lambda x: x.mean(), get_train_metrics).items()
        }
    summary['samples_per_second'] = config.batch_size / (time.time() - train_metrics_last_t)
    summary['second_per_epoch'] = (steps_per_epoch * (time.time() - train_metrics_last_t))
    summary['epoch'] = (step + 1) / steps_per_epoch

    wandb.log(summary)

    train_metrics = []
    
  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return state


# %%
# dataset_builder = tfds.builder('cifar10')
# config = defaults.get_config()
# #%%

# train_dataset = create_input_iter(dataset_builder, batch_size = config.batch_size, is_training = False, cache_dataset = config.cache_dataset)
# def create_input_iter(dataset_builder, batch_size, is_training, cache_dataset):
#   ds = data.get_dataset(dataset_builder, batch_size = config.batch_size, is_training = is_training, cache_dataset = config.cache_dataset)
#   it = map(prepare_batch, ds)
#   return it

# dataset_builder = tfds.builder('cifar10')

# ds = data.get_dataset(dataset_builder, batch_size = config.batch_size, is_training = True, cache_dataset = config.cache_dataset)

# import matplotlib.pyplot as plt

# images, labels = batch = next(iter(ds))

# p_images, p_labels = prepare_batch(batch)


# def prepare_batch(batch):
#     aug_images, labels = batch

#     local_device_count = jax.local_device_count()
#     ch = 3
    
#     # images is b × h × w × (ch * num_augs)
#     aug_images = aug_images._numpy()
#     num_augs = aug_images.shape[3] // ch
#     aug_list = jnp.split(aug_images, num_augs, axis = -1)
    
#     # aug_list is (b * num_augs) × h × w × ch
#     images = jnp.concatenate(aug_list)

#     labels = labels._numpy()
#     labels = jnp.repeat(labels, images.shape[0] // labels.shape[0], axis = 0)

#     # xla split is xla × (b * num_augs)/xla × h × w × ch
#     xla_split_images = images.reshape((local_device_count, -1) + images.shape[1:])
#     xla_split_labels = labels.reshape((local_device_count, -1) + labels.shape[1:])
#     return xla_split_images, xla_split_labels