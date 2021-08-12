# %% 1: Imports
import functools
from typing import Any

import flax
import jax
import ml_collections
import optax
import torch
import torchvision
import torchvision.transforms as transforms
from flax.training import train_state
from jax import numpy as jnp
from jax import random, lax

import defaults
import loss_functions
import models

# %% 2: Define model

def create_model(resnet="ResNet50", stem="CIFAR", half_precision=True, **kwargs):
    is_tpu_platform = jax.local_devices()[0].platform == 'tpu'
    if half_precision:
        model_dtype = jnp.bfloat16 if is_tpu_platform else jnp.float16
    else:
        model_dtype = jnp.float32

    stem = getattr(models.stem, stem)
    resnet = getattr(models.resnet, resnet)

    return resnet(stem=stem, dtype=model_dtype)

# generates model variable placeholders


def initialized(rng, image_size, model):
    input_shape = (1, 3, image_size, image_size)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init(rng, jnp.ones(input_shape, model.dtype))
    return variables['params'], variables['batch_stats']


## %% 3: Define loss
# loss_fn = functools.partial(loss_functions.ntxent, temp=0.5)

# #%% 4: Metric computation
# #
# def compute_metrics(logits, labels):
#   accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
#   metrics = {
#       'loss': loss,
#       'accuracy': accuracy,
#   }
#   metrics = lax.pmean(metrics, axis_name='batch')
#   return metrics

# %% 5: Loading data

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]
)
dataset_train = torchvision.datasets.CIFAR10(
    root="./",  train=True, download=True, transform=transform_test)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=128)
batch, targets = next(iter(loader_train))
batch = jnp.asarray(batch.numpy())

# %% 6: Create train state

class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: flax.optim.DynamicScale


def create_learning_rate_fn(
        config: ml_collections.ConfigDict,
        base_learning_rate: float,
        steps_per_epoch: int):
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch)

    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)

    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch)

    learning_rate_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch])

    return learning_rate_fn

def create_train_state(rng, config: ml_collections.ConfigDict,
                       model, image_size, learning_rate_fn) -> TrainState:
    """Create initial training state."""
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == 'gpu':
        dynamic_scale = flax.optim.DynamicScale()
    else:
        dynamic_scale = None

    params, batch_stats = initialized(rng, image_size, model)
    tx = optax.sgd(
        learning_rate=learning_rate_fn,
        momentum=config.momentum,
        nesterov=True,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=tx,
        dynamic_scale=dynamic_scale)
    return state


config = defaults.get_config()
config.warmup_epochs = 10
config.num_epochs = 1000
config.dtype = jnp.float16
learning_rate_fn = create_learning_rate_fn(config, 1.5, 100)

model = create_model()
image_size = 32
state = create_train_state(random.PRNGKey(
    0), config, model, image_size, learning_rate_fn)


# class ContrastiveTrainState(struct.PyTreeNode):
#   step: int
#   apply_fn: Callable


# %% 7: Training step

# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

def train_step(state, batch, learning_rate_fn):
  """Perform a single training step."""
  def loss_fn(params):
    """loss function used for training."""
    batch_a = batch
    encod_a, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        batch_a,
        mutable=['batch_stats'])

    encod_b = encod_a
    loss, align, unif = loss_functions.ntxent(encod_a, encod_b)
    
    return loss, align, unif, new_model_state

  step = state.step
  dynamic_scale = state.dynamic_scale
  lr = learning_rate_fn(step)

  is_fin = None
  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(
        loss_fn, has_aux=True, axis_name='batch')
    dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    print("aux: {}".format(aux))
    grads = lax.pmean(grads, axis_name='batch')

  loss, align, unif, new_model_state = aux

  loss = lax.pmean(loss, axis_name='batch')


  new_state = state.apply_gradients(
      grads=grads, batch_stats=new_model_state['batch_stats'])
  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
    new_state = new_state.replace(
        opt_state=jax.tree_multimap(
            functools.partial(jnp.where, is_fin),
            new_state.opt_state,
            state.opt_state),
        params=jax.tree_multimap(
            functools.partial(jnp.where, is_fin),
            new_state.params,
            state.params))

  return new_state, loss

#%% Test train step
config = defaults.get_config()
config.warmup_epochs = 10
config.num_epochs = 1000
config.dtype = jnp.float16
learning_rate_fn = create_learning_rate_fn(config, 1.5, 100)

model = create_model(resnet = "_ResNet1")
state = create_train_state(random.PRNGKey(0), config, model, image_size, learning_rate_fn)

p_train_step = jax.pmap(
    functools.partial(train_step, learning_rate_fn=learning_rate_fn),
    axis_name='batch')
new_state, loss = p_train_step(state, [batch])


# # #%% 8. Evaluation step

# # learning_rate = 1.5


# # optimizer = optax.sgd(learning_rate)
# # otimizer_state = optimizer.init(params)

# # updates, opt_state = optimizer.update(grads, opt_state)
# # params = optax.apply_updates(params, updates)
