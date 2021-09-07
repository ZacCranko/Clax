#%%
import os
from absl import logging

from typing import Any
from ml_collections import ConfigDict

import jax, flax, optax
from jax import random, numpy as jnp
from jax.random import PRNGKey

from flax.training import train_state
from flax.core import freeze, unfreeze
import flax.training.checkpoints as chkp

import models
from linear_evaluation_lbfgs import EncoderState


class CLTrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: flax.optim.DynamicScale

    def apply(self, x, train: bool = True):
        return self.apply_fn(self.params, x, train=train)


def create_assembly(config: ConfigDict, axis_name: str = "batch"):
    is_tpu_platform = jax.local_devices()[0].platform == "tpu"
    if config.half_precision:
        model_dtype = jnp.bfloat16 if is_tpu_platform else jnp.float16
    else:
        model_dtype = jnp.float32

    stem = getattr(models.stem, config.stem)

    backbone = getattr(models.resnet, config.model)
    backbone = backbone(stem=stem, dtype=model_dtype)

    projector = getattr(models.projector, config.projector)
    projector = projector(dtype=model_dtype, axis_name=axis_name)
    assembly = models.projector.Assembly(
        backbone=backbone, projector=projector, dtype=model_dtype
    )
    return assembly


def initialized(rng: jnp.ndarray, image_shape: int, model):
    input_shape = (128,) + image_shape

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init(rng, jnp.ones(input_shape, model.dtype))
    return variables["params"], variables["batch_stats"]


def create_learning_rate_fn(config: ConfigDict, steps_per_epoch: int):
    base_learning_rate = config.learning_rate * config.batch_size / 256.0

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
    rng, config: ConfigDict, assembly, image_shape, learning_rate_fn, workdir: str
) -> CLTrainState:
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
        tx = optax.masked(tx, freeze({"backbone": True, "projector": False}))

    state = CLTrainState.create(
        apply_fn=assembly.apply,
        params=params,
        batch_stats=batch_stats,
        tx=tx,
        dynamic_scale=dynamic_scale,
    )

    if config.restore_projector != "":
        logging.info(f"Restoring the projector parameters from {stored_run}")

        # if a full path is supplied, use that, otherwise treat it as the name of another run
        stored_run = config.restore_projector
        checkpoint_path = (
            stored_run
            if os.path.isdir(stored_run)
            else os.path.join(workdir, "..", stored_run)
        )

        stored_run_dict = chkp.restore_checkpoint(checkpoint_path, None)
        stored_projector_params = stored_run_dict["params"]["projector"]

        # sanity check to see if the stored run has the same parameters as the projector we have initialised
        if state.params["projector"].keys() != stored_projector_params.keys():
            raise ValueError(
                f"Config projector architecture ({config.projector}) does not match run {stored_run}"
            )

        params = unfreeze(state.params)
        params["projector"] = stored_projector_params

        state = state.replace(params=freeze(params))

    return state


def create_encoder_state(
    state: CLTrainState, assembly: flax.linen.Module
) -> EncoderState:
    params = flax.core.frozen_dict.freeze(
        {
            "params": state.params["backbone"],
            "batch_stats": state.batch_stats["backbone"],
        }
    )
    return EncoderState.create(apply_fn=assembly.backbone.apply, params=params)


def restore_encoder_state(
    config: ConfigDict, workdir: str, image_shape
) -> EncoderState:
    assembly = create_assembly(config)
    state = create_train_state(
        random.PRNGKey(0),
        config,
        assembly,
        image_shape,
        lambda _: 0,
        workdir,
    )
    state = chkp.restore_checkpoint(workdir, state)
    return create_encoder_state(state, assembly)
