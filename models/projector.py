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
  axis_name: str = None

  @nn.compact
  def __call__(self, x, train: bool = True):
    dense = partial(nn.Dense, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype,
                   axis_name=self.axis_name)

    for i, size in enumerate(self.stage_sizes[:-1]):
      x = dense(features=size)(x)
      x = norm()(x)
      x = self.act(x)

    x = dense(features=self.stage_sizes[-1])(x)

    return x


SimCLR = partial(MLP, stage_sizes=[2048, 128])

CIFAR10Classifier = partial(MLP, stage_sizes=[10])

Linear128 = partial(MLP, stage_sizes=[128])


class LinearSame(nn.Module):
  dtype: Any = jnp.float32
  axis_name: str = None

  @nn.compact
  def __call__(self, x, **kwargs):
    return nn.Dense(features=x.shape[-1], use_bias=False)(x)


class Identity(nn.Module):
  dtype: Any = jnp.float32
  axis_name: str = None

  @nn.compact
  def __call__(self, x, **kwargs):
    return x


class Square(nn.Module):
  dtype: Any = jnp.float32
  axis_name: str = None

  @nn.compact
  def __call__(self, x, **kwargs):
    return nn.Dense(features=x.shape[-1], use_bias=True)(x)


class Assembly(nn.Module):
  backbone: Any
  projector: Any
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train: bool = True):
    return self.projector(self.backbone(x, train=train), train=train)
