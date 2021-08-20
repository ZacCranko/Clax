from functools import partial
from typing import Any, Callable, Sequence, Tuple

ModuleDef = Any

from flax import linen as nn
import jax.numpy as jnp

class CIFAR(nn.Module):
  """CIFAR ResNet Stem"""
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  axis_name: str = None

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype,
                   axis_name=self.axis_name)

    x = conv(self.num_filters, (3, 3), (1, 1),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    return x

class ImageNet(nn.Module):
  """ImageNet ResNet Stem"""
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  axis_name: str = None

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype,
                   axis_name=self.axis_name)

    x = conv(self.num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    return x