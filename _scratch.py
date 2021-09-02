#%%
import os
##
from absl import logging 

from typing import Any
from ml_collections import ConfigDict

import jax, flax, optax
from jax import random, numpy as jnp
from jax.random import PRNGKey

from flax.training import train_state
import flax.training.checkpoints as chkp

import models, defaults, init 

config = defaults.get_config()
config.freeze_projector = True
config.restore_projector = "morning-river-67"
##

rng = PRNGKey(0)

dynamic_scale = None
platform = jax.local_devices()[0].platform
if config.half_precision and platform == "gpu":
    dynamic_scale = flax.optim.DynamicScale()
else:
    dynamic_scale = None

image_shape = (32,32,3)
assembly = init.create_assembly(config)
params, batch_stats = init.initialized(rng, image_shape, assembly)
learning_rate_fn = init.create_learning_rate_fn(config, 200)
tx = optax.lars(
    learning_rate=learning_rate_fn,
    momentum=config.momentum,
    nesterov=True,
)

if config.freeze_projector:
    tx = optax.masked(tx, flax.core.freeze({"backbone" : True, "projector" : False}))


##
config = defaults.get_config()
init.restore_encoder_state(config, os.path.join("checkpoints", "morning-river-67"))
##
workdir = "checkpoints"
stored_run = config.restore_projector
stored_run_dict = chkp.restore_checkpoint(os.path.join(workdir, stored_run), None)

stored_projector_params = stored_run_dict['params']['projector']
if state.params['projector'].keys() != stored_run_dict['params']['projector'].keys():
  raise ValueError(f"Config projector architecture ({config.projector}) does not match run {stored_run}")

params = flax.core.unfreeze(state.params)
params['projector'] = stored_projector_params

state.replace(params = flax.core.freeze(params))

##
if config.restore_projector != "":
    run = config.restore_projector

    logging.info(f"Restoring the projector parameters from {run}")
    stored_run_dict = chkp.restore_checkpoint(os.path.join(workdir, stored_run), None)

    stored_projector_params = stored_run_dict['params']['projector']

    if state.params['projector'].keys() != stored_projector_params.keys():
      raise ValueError(f"Config projector architecture ({config.projector}) does not match run {stored_run}")

    params = flax.core.unfreeze(state.params)
    params['projector'] = stored_projector_params

    state = state.replace(params = flax.core.freeze(params))
