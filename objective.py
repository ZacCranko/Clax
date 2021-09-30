import jax, functools
from jax.numpy import ndarray
from jax import random, numpy as jnp, lax, tree_util
import jax.ops
from jax.scipy.special import logsumexp

from collections import namedtuple
import optax._src.numerics as numerics
import optax


# doesn't seem to be a built-in implementation of this
@jax.jit
def normalize(input, min_norm: float = 1e-4):
  factor = jnp.linalg.norm(input, ord=2, axis=-1, keepdims=True)
  factor = jnp.maximum(factor, min_norm)
  return input / factor


@jax.jit
def ntxent_single(encodings, temp: float = 0.5, eps: float = 1e-4):
  """Normalised Temperature Cross Entropy"""
  # use this to verify correctness of the lower implementation
  encodings = normalize(encodings, min_norm=eps)
  encod_a, encod_b = jnp.array_split(
      jnp.reshape(encodings, (-1, encodings.shape[-1])), 2)

  # cross correlation matrices
  xcor_aa = jnp.matmul(encod_a, encod_a.T) / temp
  xcor_bb = jnp.matmul(encod_b, encod_b.T) / temp
  xcor_ab = jnp.matmul(encod_a, encod_b.T) / temp

  # smaller numbers here means the pairs are more aligned
  align = -jnp.diag(xcor_ab).mean()

  # need to exclude some elements from the logsumexp
  xcor_aa = jax.ops.index_update(xcor_aa, jnp.diag_indices_from(xcor_aa),
                                 -jnp.inf)
  xcor_bb = jax.ops.index_update(xcor_bb, jnp.diag_indices_from(xcor_bb),
                                 -jnp.inf)

  cat_a = jnp.concatenate((xcor_aa, xcor_ab.T), axis=-1)
  cat_b = jnp.concatenate((xcor_ab, xcor_bb), axis=-1)

  # smaller numbers here means the pairs are more uniform
  unif = logsumexp(cat_a, axis=-1).mean() / 2 + logsumexp(cat_b,
                                                          axis=-1).mean() / 2
  loss = align + unif

  return loss, (-align * temp, -unif * temp)


def ntxent(
    device_id: int,
    batch: ndarray,
    temp: float = 0.5,
    unif_coeff: float = 1.0,
    axis_name: str = "batch",
    min_norm: float = 1e-4,
):
  batch = normalize(batch, min_norm=min_norm)
  full_batch_by_device = jax.lax.all_gather(batch, axis_name=axis_name)

  num_xla, batch_size, _ = full_batch_by_device.shape
  full_batch = jnp.reshape(full_batch_by_device,
                           (-1, full_batch_by_device.shape[-1]))

  pair_index = (device_id + num_xla // 2) % num_xla

  align = -jnp.sum(jnp.multiply(batch, full_batch_by_device[pair_index])) / (
      temp * batch_size)

  # psuedo cross corrrelation sub-matrix
  xcor = batch @ full_batch.transpose() / temp

  # diag_inds are the indices of the diagonal of the sub-matrix
  # batch * batch' within xcor, need to set the diagonal to -inf to exclude from logsumexp
  row_inds = jnp.arange(batch_size)
  diag_inds = (row_inds, row_inds + device_id * batch_size)
  xcor_corrected = jax.ops.index_update(xcor, diag_inds, -jnp.inf)

  unif = logsumexp(xcor_corrected, axis=-1).mean()
  loss = align + unif_coeff * unif

  return loss, (-align * temp, -unif * temp)


@jax.jit
def accuracy(*, logits, labels):
  return jnp.argmax(logits, -1) == jnp.argmax(labels, -1)


@jax.jit
def classification_metrics(*, logits, labels):
  metrics = {
      "cross_entropy":
          optax.softmax_cross_entropy(logits=logits, labels=labels).mean(),
      "accuracy":
          jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1)),
  }
  return metrics