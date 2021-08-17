# %% 1: Imports
import os, sys, time
from typing import Any
from absl import logging

sys.path.append("../simclr")
from simclr_iterator import build_input_fn

import jax, flax, optax, ml_collections, functools

import torchvision.transforms as transforms
from flax.training import common_utils, train_state
from jax import numpy as jnp, random, lax
from flax import jax_utils

import tensorflow as tf
import tensorflow_datasets as tfds

import defaults, loss_functions, models 

# %% 2: Define model
def create_model(resnet="ResNet50", stem="CIFAR", half_precision=False, **kwargs):
    is_tpu_platform = jax.local_devices()[0].platform == "tpu"
    if half_precision:
        model_dtype = jnp.bfloat16 if is_tpu_platform else jnp.float16
    else:
        model_dtype = jnp.float32

    stem = getattr(models.stem, stem)
    resnet = getattr(models.resnet, resnet)

    return resnet(stem=stem, dtype=model_dtype)

# generates model variable placeholders
def initialized(rng, image_size, model):
    input_shape = (128, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init(rng, jnp.ones(input_shape, model.dtype))
    return variables["params"], variables["batch_stats"]

# %% 5: Loading data

def create_input_iter(dataset_builder, global_batch_size: int = 128, split: str = 'train', is_training: bool = True):
  it = build_input_fn(dataset_builder, global_batch_size, is_training, train_mode = 'pretrain', split = split)()
  return it

def prepare_batch(batch):
    aug_images, labels = batch

    local_device_count = jax.local_device_count()
    ch = 3
    
    # images is b × h × w × (ch * num_augs)
    aug_images = aug_images._numpy()
    labels = labels._numpy()

    num_augs = aug_images.shape[3] // ch
    aug_list = jnp.split(aug_images, num_augs, axis = -1)
    
    # aug_list is (b * num_augs) × h × w × ch
    images = jnp.concatenate(aug_list)

    # spmd split is xla × (b * num_augs)/xla × h × w × ch
    spmd_split_images = images.reshape((local_device_count, -1) + images.shape[1:])
    spmd_split_labels = labels.reshape((local_device_count, -1) + labels.shape[1:])
    return spmd_split_images, spmd_split_labels

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
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
) -> TrainState:
    """Create initial training state."""
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == "gpu":
        dynamic_scale = flax.optim.DynamicScale()
    else:
        dynamic_scale = None

    params, batch_stats = initialized(rng, image_size, model)
    tx = optax.lamb(
        learning_rate=learning_rate_fn,
        # momentum=config.momentum,
        # nesterov=True,
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

def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def train_step(state, device_id, batch, learning_rate_fn):
    """Perform a single training step."""
    """Expects to run in a parallel space with an axis name 'device'"""

    def loss_fn(params):
        """loss function used for training."""
        encodings, new_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch, mutable=["batch_stats"],
        )
        loss, (align, unif) = loss_functions.ntxent(device_id, encodings, temp = 0.5)

        return loss, (new_state, align, unif)

    step = state.step
    metrics = dict()
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_state, align, unif)), grads = grad_fn(state.params)

    grads = lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_state["batch_stats"]
    )

    metrics["learning_rate"] = learning_rate_fn(step)
    metrics["loss"]  = lax.pmean(loss, axis_name='batch')
    metrics["align"] = lax.pmean(align, axis_name='batch')
    metrics["unif"]  = lax.pmean(unif, axis_name='batch')
    return new_state, metrics

#%%

def train_and_evaluate(config: ml_collections.ConfigDict) -> TrainState:
  rng = random.PRNGKey(0)

  image_size = 32

  dataset_builder = tfds.builder(config.dataset)
  num_examples = dataset_builder.info.splits['train'].num_examples
  steps_per_epoch = num_examples // config.batch_size 

  train_iter = create_input_iter(dataset_builder, global_batch_size = config.batch_size)

  base_learning_rate = config.learning_rate * config.batch_size / 256.
  model = create_model(config.model, config.stem)
  learning_rate_fn = create_learning_rate_fn(
      config, base_learning_rate, steps_per_epoch)
  state = create_train_state(
      random.PRNGKey(0), config, model, image_size, learning_rate_fn
  )
  step_offset = int(state.step)

  state = jax_utils.replicate(state)

  device_ids = jnp.arange(jax.local_device_count())
  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn = learning_rate_fn),
      axis_name="batch")

  # batch  = next(iter(train_iter))
  # image_global_batch, labels_batch = prepare_batch(batch)
  # state, loss = p_train_step(state, device_ids, image_global_batch)
  
  train_metrics = []
  num_steps = int(steps_per_epoch * config.num_epochs)
  train_metrics_last_t = time.time()
  logging.info('Initial compilation.')
  for step, batch in zip(range(step_offset, num_steps), train_iter):
    image_batch, _ = prepare_batch(batch)
    state, metrics = p_train_step(state, device_ids, image_batch)
    if step == step_offset:
      logging.info('Initial compilation completed.')
    train_metrics.append(metrics)
    if (step + 1) % config.log_every_steps == 0:
      get_train_metrics = common_utils.get_metrics(train_metrics)
      summary = {
              f'train_{k}': v
              for k, v in jax.tree_map(lambda x: x.mean(), get_train_metrics).items()
          }
      summary['steps_per_second'] = config.log_every_steps / (time.time() - train_metrics_last_t)
      summary['epoch'] = (step + 1) / steps_per_epoch
      wandb.log(summary)

      logging.info('epoch: %.2f, loss: %.4f, align: %.2, unif: %.2', 
        epoch, summary["train_loss"], summary["train_align"], summary["train_unif"])

      train_metrics = []
      train_metrics_last_t = time.time()

    # if (step + 1) % steps_per_epoch == 0:
      # epoch = step//steps_per_epoch
      # logging.info('eval epoch: %d, loss: %.4f', epoch, jnp.mean(loss))

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return state

#%%
import wandb

wandb.init(project='jax', entity='zaccranko')
config = defaults.get_config()
logging.set_verbosity(logging.INFO)
train_and_evaluate(config)
#%%

# p_train_step = jax.pmap(
#     functools.partial(train_step, learning_rate_fn=learning_rate_fn),
#     axis_name='batch')

# state = jax_utils.replicate(state)
# device_ids = jnp.arange(jax.device_count())
# # %%
# state, loss = p_train_step(state, device_ids, image_global_batch)



# p_train_step()


# def loss_fn(params):
#     """loss function used for training."""
#     encodings, new_model_state = state.apply_fn(
#         {"params": params, "batch_stats": state.batch_stats},
#         image_batch[0],
#         mutable=["batch_stats"],
#     )

#     loss, aux = loss_functions.pytorch_ported_ntxent(encodings, temp = 0.5)

#     return loss, (new_model_state, aux)

# grad_fn = jax.value_and_grad(loss_fn, has_aux=True)


# aux, grads = grad_fn(state.params)

# @jax.pmap
# def forward_pass(images):
#   encod = state.apply_fn({"params": state.params, "batch_stats": state.batch_stats}, images, train = False)

#   return encod

# encodings = forward_pass(images)
# loss, align, unif = loss_functions.ntxent(encodings)


# it = create_input_iter(tfds.builder("cifar10"), global_batch_size= 1024)


# num_transforms = inputs.shape[3] // 3
# num_transforms = tf.repeat(3, num_transforms)
# # Split channels, and optionally apply extra batched augmentation.
# features_list = tf.split(
#     features, num_or_size_splits=num_transforms, axis=-1)
# if FLAGS.use_blur and training and FLAGS.train_mode == 'pretrain':
#   features_list = data_util.batch_random_blur(features_list,
#                                               FLAGS.image_size,
#                                               FLAGS.image_size)
# features = tf.concat(features_list, 0)  # (num_transforms * bsz, h, w, c)
