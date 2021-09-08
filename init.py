#%%
import os
from absl import logging

from typing import Any, Callable, Tuple
from ml_collections import ConfigDict

import jax, flax, optax
from jax import random, numpy as jnp
from flax.linen import Module
from jax.numpy import ndarray

from flax.training import train_state
from flax.core import freeze, unfreeze
from flax import struct
import flax.training.checkpoints as chkp

import models


class CLTrainState(train_state.TrainState):
    backbone_apply_fn: Callable = struct.field(pytree_node=False)
    batch_stats: Any
    
    clf_heads_apply_fn: Callable = struct.field(pytree_node=False)
    clf_heads_params: Any

    def encode(self, x, train: bool = False):
        params = dict(params=self.params["backbone"], batch_stats = self.batch_stats["backbone"])
        return self.backbone_apply_fn(params, x, train = train, mutable=False)

    def classify(self, encodings, train: bool = False):
        return self.clf_heads_apply_fn(dict(params=self.clf_heads_params), encodings)

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


def initialise_assembly(key: jnp.ndarray, image_shape: Tuple[int,int,int], model: Module):
    input_shape = (128,) + image_shape

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init(key, jnp.ones(input_shape, model.dtype))
    return variables


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

def initialise_linear_clf_heads(
    key: ndarray, input_dim: int, num_classes: int, num_heads: int = 10
):
    clf = models.classifier.MutiHeadClassifier(
        num_heads=num_heads, num_classes=num_classes
    )
    params = clf.init(key, jnp.ones((0, input_dim)))
    return clf.apply, params

def create_train_state(
    key, config: ConfigDict, assembly, image_shape,learning_rate_fn, workdir: str
) -> CLTrainState:
    """Create initial training state."""
    # dynamic_scale = Noneå
    # platform = jax.local_devices()[0].platform

    assembly_params = initialise_assembly(key, image_shape, assembly)

    clf_heads_apply_fn, clf_heads_params = initialise_linear_clf_heads(key, input_dim = 2048, num_classes = 10, num_heads = 1)

    tx = optax.lars(
        learning_rate=learning_rate_fn,
        momentum=config.momentum,
        nesterov=True,
    )

    if config.freeze_projector:
        tx = optax.masked(tx, freeze({"backbone": True, "projector": False}))

    state = CLTrainState.create(
        apply_fn=assembly.apply,
        backbone_apply_fn = assembly.backbone.apply,
        params=assembly_params["params"],
        batch_stats=assembly_params["batch_stats"],
        tx=tx,
        clf_heads_apply_fn = clf_heads_apply_fn,
        clf_heads_params = clf_heads_params['params']
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

        assembly_params = unfreeze(state.params)
        assembly_params["projector"] = stored_projector_params

        state = state.replace(params=freeze(assembly_params))

    return state


# def create_encoder_state(
#     state: CLTrainState, assembly: flax.linen.Module
# ) -> EncoderState:
#     params = flax.core.frozen_dict.freeze(
#         {
#             "params": state.params["backbone"],
#             "batch_stats": state.batch_stats["backbone"],
#         }
#     )
#     return EncoderState.create(apply_fn=assembly.backbone.apply, params=params)


# def restore_encoder_state(
#     config: ConfigDict, workdir: str, image_shape
# ) -> EncoderState:
#     assembly = create_assembly(config)
#     state = create_train_state(
#         random.PRNGKey(0),
#         config,
#         assembly,
#         image_shape,
#         lambda _: 0,
#         workdir,
#     )
#     state = chkp.restore_checkpoint(workdir, state)
#     return create_encoder_state(state, assembly)
