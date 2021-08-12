import flax.linen as nn
from functools import partial

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any

class MLP(nn.Module):
  """Multi-Layer Perceptron"""
  stage_sizes: Sequence[int]
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  
  @nn.compact
  def __call__(self, x, train: bool = True):
    dense = partial(nn.Dense, use_bias = False, dtype = self.dtype)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype)

    for i, size in enumerate(self.stage_sizes[:-1]):
        x = dense(features = size)(x)
        x = norm()(x)
        x = self.act(x)

    x = dense(features = self.stage_sizes[-1])(x)
    
    return x

SimCLR = partial(MLP, stage_sizes=[2048, 2048, 128])