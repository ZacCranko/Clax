from functools import partial
from typing import Any, Callable, Sequence, Tuple

ModuleDef = Any

from flax import linen as nn
import jax.numpy as jnp

class CIFAR(nn.Module):
  """CIFAR ResNet Stem"""
  norm: ModuleDef
  conv: ModuleDef
  num_filters: int = 64

  @nn.compact
  def __call__(self, x, train: bool = True):
    x = self.conv(self.num_filters, (3, 3), (1, 1),
                  padding=[(3, 3), (3, 3)],
                  name='conv_stem')(x)
    x = self.norm(name='bn_stem')(x)
    x = nn.relu(x)
    return x

class ImageNet(nn.Module):
  """ImageNet ResNet Stem"""
  norm: ModuleDef
  conv: ModuleDef
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu

  @nn.compact
  def __call__(self, x, train: bool = True):
    x = self.conv(self.num_filters, (7, 7), (2, 2),
                  padding=[(3, 3), (3, 3)],
                  name='conv_stem')(x)
    x = self.norm(name='bn_stem')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    return x