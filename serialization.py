#%%
import jax, flax, ml_collections
import flax.training.checkpoints as checkpoints
from absl import logging

def save_projector(config: ml_collections.ConfigDict, state, desc = None, step: int = -1):
  projector_params = state.params['projector']
  
  name_components = [config.projector]
  if desc is not None:
    name_components.append(desc)

  prefix = "-".join(name_components)
  checkpoints.save_checkpoint(ckpt_dir = "checkpoints/projector",
                              target = projector_params,
                              step = step,
                              prefix = prefix + "_",
                              keep = 1, overwrite=True)

def restore_projector(config: ml_collections.ConfigDict, state, desc = None):
  logging.info("Restoring the projector parameters")
  projector_params = state.params['projector']
  
  name_components = [config.projector]
  if desc is not None:
    name_components.append(desc)

  prefix = "-".join(name_components)
  new_projector_params = checkpoints.restore_checkpoint(ckpt_dir = "checkpoints/projector", 
                                                        target = projector_params,
                                                        prefix = prefix + "_")
  params = flax.core.unfreeze(state.params)
  params['projector'] = flax.core.unfreeze(new_projector_params)
  params = flax.core.freeze(params)
  return state.replace(params = params)
                            
# if config.restore_projector:
    # state = serialization.restore_projector(config, state = state)

# return state

# import defaults, init
# from jax import random
# config = defaults.get_config()
# assembly = init.create_assembly(config)
# config.
# state = init.create_train_state(random.PRNGKey(0), config, assembly, (32, 32 ,3), 0)
