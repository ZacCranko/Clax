import flax.linen as nn
from functools import partial

from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

class MutiHeadClassifier(nn.Module):
  num_heads: int 
  num_classes: int 
  dtype: Any = jnp.float32
  
  @nn.compact
  def __call__(self, x):
    clf = partial(nn.Dense, features = self.num_classes, dtype = self.dtype)
    clfs = [ clf(name = f"head_{head}")(x) for head in range(self.num_heads) ]
    return jnp.stack(clfs)
  