# %%
import jax, functools
from jax import random, numpy as jnp, lax
from jax.scipy.special import logsumexp

from collections import namedtuple

# doesn't seem to be a built-in implementation of this
@jax.jit
def normalize(input, ord = 2, eps = 1e-6):
  factor = jnp.linalg.norm(input, ord = ord, axis = -1, keepdims=True)
  factor = jnp.maximum(factor, eps)
  return input / factor

@jax.jit
def _ntxent(encod_a, encod_b, temp = 0.5, eps = 1e-6):
  """Normalised Temperature Cross Entropy"""
  # use this to verify correctness of the lower implementation
  encod_a = normalize(encod_a, eps = eps)
  encod_b = normalize(encod_b, eps = eps)

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
  unif = logsumexp(cat_a, axis = -1).mean() + logsumexp(cat_b, axis = -1).mean()
  unit = unif/2

  return align + unif, align, unif


@functools.partial(jax.pmap, axis_name = "device")
def ntxent_spmd(device_id, batch, temp):
  batch = normalize(batch)
  all_batches = jax.lax.all_gather(batch, axis_name = "device")
  full_batch = jnp.reshape(all_batches, (-1, all_batches.shape[-1]))

  # cross corrrelation sub-matrix
  xcor = batch @ full_batch.transpose() / temp

  # this is some garbage we have to do because jax sucks at tracing off-diagonal ranges
  # and I can't figure out how to call matmul when all_batches is 3-dimensional
  index = (device_id + 4) % all_batches.shape[0]
  align = -jnp.mean(batch @ all_batches[index].transpose() / temp)

  # diag_inds are the indices of the diagonal of the sub-matrix 
  # batch * batch' within xcor
  row_inds = jnp.arange(batch.shape[0])
  diag_inds = (row_inds, row_inds + device_id)
  xcor_corrected = jax.ops.index_update(xcor, diag_inds, -jnp.inf)
  
  unif = jax.nn.logsumexp(xcor_corrected, axis = -1) * 2
  loss = - (align + unif)

  return loss, (align, unif)

#%%
# batch_size = 128
# encoding_dim = 2048
# encodings = random.uniform(random.PRNGKey(20), (jax.device_count(), batch_size, encoding_dim), dtype=jnp.bfloat16)

# encodings_a, encodings_b = jnp.array_split(jnp.reshape(encodings, (-1, encodings.shape[-1])), 2)


# print("Single NTXEnt")
# %timeit -n 100 single = _ntxent(encodings_a, encodings_b, temp = 5)
# print([float(x) for x in single])


# print("Parallel NTXEnt")
# %timeit -n 100 parallel = ntxent(encodings, temp = 5)
# print([float(x) for x in parallel])

# batch = full_batch[0]
# id = 0

# batch_size, _ = batch.shape
# device_id = 1
# diag_indices = (jnp.arange(batch_sze), jnp.arange(batch_size) + device_id)

# batch = full_batch[0]

# # jnp.arange()

# # jax.ops.index_update(batch, , -jnp.inf)
# # %%


# row_inds = jnp.arange(1024)
# diag_inds = (row_inds, row_inds + device_id)
