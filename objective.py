from typing import Tuple
import jax, functools
from jax.numpy import ndarray
from jax import random, numpy as jnp, lax, tree_util
import jax.ops
from jax.scipy.special import logsumexp

import optax


# doesn't seem to be a built-in implementation of this
@jax.jit
def normalize(input: ndarray, min_norm: float = 1e-4) -> ndarray:
  """Normalise a batch to have unit norm

  Args:
      input (ndarray): Batch to be normalised along its last axis
      min_norm (float, optional): Numerical stability parameter. Defaults to 1e-4.

  Returns:
      ndarray: [description]
  """
  factor = jnp.linalg.norm(input, ord=2, axis=-1, keepdims=True)
  factor = jnp.maximum(factor, min_norm)
  return input / factor


def ntxent(
    device_id: int,
    batch: ndarray,
    temp: float = 0.5,
    unif_coeff: float = 1.0,
    axis_name: str = "batch",
    min_norm: float = 1e-4,
) -> Tuple[float, Tuple[float, float]]:
  """[summary]

  Args:
    device_id (int): XLA device number this function is excuting on
      batch (ndarray): Batch of encodings
      temp (float, optional): Cross entropy temperature parameter. Defaults to 0.5.
      unif_coeff (float, optional): Coefficient on the uniformity term. Defaults to 1.0.
      axis_name (str, optional): Used for reference by the outer `pmap` function. Defaults to "batch".
      min_norm (float, optional): Numerical stability parameter for the normalisation. Defaults to 1e-4.

      In the return values of alignment and uniformity, greater values indicate greater alignment/uniformity.

  Returns:
      Tuple[float, Tuple[float, float]]: Returns a tuple of (ntxent, (alignment, uniformity))
  """
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