# %%
import jax, functools
from jax import random, numpy as jnp, lax
import jax.ops
from jax.scipy.special import logsumexp

from collections import namedtuple
import optax._src.numerics as numerics
# doesn't seem to be a built-in implementation of this
@jax.jit
def normalize(input, eps: float = 1e-4):
  factor = jnp.linalg.norm(input, ord = 2, axis = -1, keepdims=True)
  factor = jnp.maximum(factor, eps)
  return input / factor

@jax.jit
def pytorch_ported_ntxent(encodings, temp: float = 0.5, eps: float = 1e-4):
  """Normalised Temperature Cross Entropy"""
  # use this to verify correctness of the lower implementation
  encodings = normalize(encodings, eps = eps)
  encod_a, encod_b = jnp.array_split(jnp.reshape(encodings, (-1, encodings.shape[-1])), 2)

  # cross correlation matrices
  xcor_aa = jnp.matmul(encod_a, encod_a.T) / temp
  xcor_bb = jnp.matmul(encod_b, encod_b.T) / temp
  xcor_ab = jnp.matmul(encod_a, encod_b.T) / temp

  # smaller numbers here means the pairs are more aligned
  align = -jnp.diag(xcor_ab).mean()

  # need to exclude some elements from the logsumexp
  xcor_aa = jax.ops.index_update(xcor_aa, jnp.diag_indices_from(xcor_aa), -jnp.inf)
  xcor_bb = jax.ops.index_update(xcor_bb, jnp.diag_indices_from(xcor_bb), -jnp.inf)

  cat_a = jnp.concatenate((xcor_aa, xcor_ab.T), axis = -1)
  cat_b = jnp.concatenate((xcor_ab, xcor_bb),   axis = -1)

  # smaller numbers here means the pairs are more uniform
  unif = logsumexp(cat_a, axis = -1).mean()/2 + logsumexp(cat_b, axis = -1).mean()/2
  loss = align + unif
  
  return loss, (-align * temp, -unif * temp)


def ntxent(device_id: int, batch, temp: float, eps: float = 1e-4):
  batch = normalize(batch, eps = eps)
  batch_by_device = jax.lax.all_gather(batch, axis_name = "batch")
  num_xla, num_rows, _ = batch_by_device.shape
  batch_by_device = jnp.reshape(batch_by_device, (-1, batch_by_device.shape[-1]))

  # psuedo cross corrrelation sub-matrix
  xcor = batch @ batch_by_device.transpose() / temp

  # this is some garbage we have to do because jax sucks at tracing off-diagonal ranges
  # and I can't figure out how to call matmul when batch_by_device is 3-dimensional
  xcor_by_device = jnp.reshape(xcor, (num_xla, num_rows, num_rows))
  index = (device_id + num_xla//2) % num_xla
  barlow_outer_product = xcor_by_device[index]
  align = -jnp.diagonal(barlow_outer_product).mean()

  # diag_inds are the indices of the diagonal of the sub-matrix 
  # batch * batch' within xcor, need to set the diagonal to -inf to exclude from logsumexp
  row_inds = jnp.arange(batch.shape[0])
  diag_inds = (row_inds, row_inds + device_id * num_rows)
  xcor_corrected = jax.ops.index_update(xcor, diag_inds, -jnp.inf)
  
  unif = logsumexp(xcor_corrected, axis = -1).mean()
  loss = align + unif

  return loss, (-align * temp, -unif * temp)

@jax.jit
def p_ntxent(global_batch, temp = 0.5):
  devices = global_batch.shape[0]
  ntxent = jax.pmap(ntxent, axis_name = "batch")
  loss, (align, unif) = ntxent(jnp.arange(devices), global_batch, jnp.repeat(temp, devices))

  return jnp.mean(loss), (jnp.mean(align), jnp.mean(unif))