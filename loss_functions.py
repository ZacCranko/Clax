# %%
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp

# doesn't seem to be a built-in implementation of this
@jax.jit
def norm_projection(input, ord = 2, eps = 1e-6):
  factor = jnp.linalg.norm(input, ord = ord, axis = -1, keepdims=True)
  factor = jnp.maximum(factor, eps)
  return input / factor 

@jax.jit
def ntxent(encod_a, encod_b, temp = 0.5, eps = 1e-6):
  """Normalised Temperature Cross Entropy"""
  encod_a = norm_projection(encod_a, eps = eps)
  encod_b = norm_projection(encod_b, eps = eps)

  # cross correlation matrices
  xcor_aa = jnp.matmul(encod_a, encod_a.T) / temp
  xcor_bb = jnp.matmul(encod_b, encod_b.T) / temp
  xcor_ab = jnp.matmul(encod_a, encod_b.T) / temp

  # smaller numbers here means the pairs are more aligned
  align = -jnp.diag(xcor_ab).mean()

  # need to exclude some elements from the logsumexp
  xcor_aa = jax.ops.index_update(xcor_aa, jnp.diag_indices(xcor_aa.shape[0]), -jnp.inf)
  xcor_bb = jax.ops.index_update(xcor_bb, jnp.diag_indices(xcor_bb.shape[0]), -jnp.inf)

  cat_a = jnp.concatenate((xcor_aa, xcor_ab.T), axis = -1)
  cat_b = jnp.concatenate((xcor_ab, xcor_bb),   axis = -1)

  # smaller numbers here means the pairs are more uniform
  unif = logsumexp(cat_a, axis = -1).mean() + logsumexp(cat_b, axis = -1).mean()
  unit = unif/2

  return align + unif, align, unif