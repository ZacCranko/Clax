# %% 1: Imports
import os,sys,functools
from typing import Any

from jax._src.api import pmap

sys.path.append("../simclr")
from simclr_iterator import build_input_fn

import jax, flax, optax, ml_collections

import torchvision.transforms as transforms
from flax.training import train_state
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

# %% 6: Create train state
class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: flax.optim.DynamicScale

    def variables(self):
      {'params': self.params, 'batch_stats': self.batch_stats}

    def apply(self, batch, train: bool = True):
      return self.apply_fn(self.variables(), batch, train = train)



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
        dynamic_scale=dynamic_scale,
    )
    return state


# class ContrastiveTrainState(struct.PyTreeNode):
#   step: int
#   apply_fn: Callable


# %% 7: Training step

# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")

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
            {"params": params, "batch_stats": state.batch_stats},
            batch_a,
            mutable=["batch_stats"],
        )

        encod_b = encod_a
        loss, align, unif = loss_functions.ntxent(encod_a, encod_b)

        return loss, align, unif, new_model_state

    step = state.step
    dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step)

    is_fin = None
    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name="batch")
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        print("aux: {}".format(aux))
        grads = lax.pmean(grads, axis_name="batch")

    loss, align, unif, new_model_state = aux

    loss = lax.pmean(loss, axis_name="batch")

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )
    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        new_state = new_state.replace(
            opt_state=jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state,
            ),
            params=jax.tree_multimap(
                functools.partial(jnp.where, is_fin), new_state.params, state.params
            ),
        )

    return new_state, loss


# %%

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

# %%

config = defaults.get_config()
config.warmup_epochs = 10
config.num_epochs = 1000
config.dtype = jnp.float16
learning_rate_fn = create_learning_rate_fn(config, 1.5, 100)

model = create_model()
image_size = 32
state = create_train_state(
    random.PRNGKey(0), config, model, image_size, learning_rate_fn
)

it = create_input_iter(tfds.builder("cifar10"), global_batch_size= 512)

batch  = next(iter(it))

images, labels = prepare_batch(batch)

@jax.pmap
def forward_pass(images):
  encod = state.apply_fn({"params": state.params, "batch_stats": state.batch_stats}, images, train = False)

  return encod

encodings = forward_pass(images)
loss, align, unif = loss_functions.ntxent(encodings)


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

