from typing import Tuple
import jax.random as random, flax, ml_collections

from ml_collections import ConfigDict
from flax.linen import Module
from data import TrainIterator
import flax.training.checkpoints as chkp
import init
# from init import create_assembly, create_train_state, create_learning_rate_fn


def create_train_state(config: ConfigDict, steps_per_epoch: int,
                       image_shape: Tuple[int, int, int]):
  assembly = init.create_assembly(config)
  learning_rate_fn = init.create_learning_rate_fn(config, steps_per_epoch)
  state = init.create_train_state(random.PRNGKey(0), config, assembly,
                                  image_shape, learning_rate_fn)
  return assembly, state


def restore_checkpoint(config: ConfigDict,
                       dir: str,
                       train_iter: TrainIterator,
                       step: int = None,
                       _debug=False):
  assembly, state = create_train_state(config, train_iter.steps_per_epoch,
                                       train_iter.image_shape)

  if _debug:
    return assembly, state

  state = chkp.restore_checkpoint(dir, state, step=step)

  return assembly, state
