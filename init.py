#%%

import jax, flax, optax, ml_collections, functools

from jax import random, numpy as jnp
from flax.training import train_state
from typing import Any, Callable

import models, serialization

class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: flax.optim.DynamicScale

def create_assembly(config: ml_collections.ConfigDict, axis_name: str = "batch"):
    is_tpu_platform = jax.local_devices()[0].platform == "tpu"
    if config.half_precision:
        model_dtype = jnp.bfloat16 if is_tpu_platform else jnp.float16
    else:
        model_dtype = jnp.float32
    
    stem = getattr(models.stem, config.stem)
    
    backbone = getattr(models.resnet, config.model)
    backbone = backbone(stem = stem, dtype = model_dtype)
    
    projector = getattr(models.projector, config.projector)
    projector = projector(dtype = model_dtype, axis_name = axis_name)
    assembly = models.projector.Assembly(backbone = backbone,
                                         projector = projector,
                                         dtype = model_dtype)
    return assembly

def initialized(rng: random.PRNGKey, image_shape: int, model):
    input_shape = (128, ) + image_shape

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init(rng, jnp.ones(input_shape, model.dtype))
    return variables["params"], variables["batch_stats"]

def create_learning_rate_fn(
    config: ml_collections.ConfigDict, steps_per_epoch: int
):
    base_learning_rate = config.learning_rate * config.batch_size / 256.

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
    rng, config: ml_collections.ConfigDict, assembly, image_shape, learning_rate_fn
) -> TrainState:
    """Create initial training state."""
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == "gpu":
        dynamic_scale = flax.optim.DynamicScale()
    else:
        dynamic_scale = None

    params, batch_stats = initialized(rng, image_shape, assembly)

    tx = optax.lars(
        learning_rate=learning_rate_fn,
        momentum=config.momentum,
        nesterov=True,
    )

    if config.freeze_projector:
        tx = optax.masked(tx, flax.core.freeze({"backbone" : False, "projector" : True}))

    state = TrainState.create(
        apply_fn=assembly.apply,
        params=params,
        batch_stats=batch_stats,
        tx=tx,
        dynamic_scale=dynamic_scale,
    )

    if config.restore_projector:
        state = serialization.restore_projector(config, state = state)

    return state

# import defaults 
# config = defaults.get_config()
# config.restore_projector = False
# config.freeze_projector = True

# assembly = create_assembly(config)
# state = create_train_state(random.PRNGKey(0), config, assembly, (32, 32 ,3), 0)