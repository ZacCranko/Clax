#%%
from absl.testing import absltest
import sys
sys.path.append("..")
# sys.path.append("../../contrastive")
import objective
import torch_loss
import numpy as np
import jax
from jax import random, numpy as jnp
import torch 
from torch import nn
import functools

import optax._src.numerics as numerics

_p_ntxent = jax.pmap(functools.partial(objective.ntxent, axis_name = "batch"), axis_name = "batch")

def p_ntxent(global_batch, temp = 0.5):
  devices = global_batch.shape[0]
  loss, (align, unif) = _p_ntxent(jnp.arange(devices), global_batch, jnp.repeat(temp, devices))

  return jnp.mean(loss), (jnp.mean(align), jnp.mean(unif))
  
batch_size = 4
encoding_dim = 4
temp = 0.5
key = random.PRNGKey(0)

for temp in jnp.linspace(0.1, 2, 5):
  key, subkey = random.split(key)
  encodings = random.uniform(subkey, (jax.device_count(), batch_size, encoding_dim), dtype=jnp.float32)
  encod_a, encod_b = jnp.array_split(jnp.reshape(encodings, (-1, encodings.shape[-1])), 2)

  encod_a, encod_b = torch.tensor(np.array(encod_a)), torch.tensor(np.array(encod_b))
  pytorch_ntxent = torch_loss.NTXEntLoss(temperature = temp)
  t_loss = pytorch_ntxent(encod_a, encod_b)
  t_loss = t_loss.item()
  (t_align, t_unif) = (pytorch_ntxent.alignment.item(), pytorch_ntxent.uniformity.item() * temp)

  s_loss, (s_align, s_unif) = objective.pytorch_ported_ntxent(encodings, temp = temp)
  p_loss, (p_align, p_unif) = p_ntxent(encodings, temp = temp)

  print(f"temp: {temp}")
  print(f"t_loss {t_loss:.10f}\t t_align: {t_align:.10f}\t t_unif {t_unif:.10f}\t")
  print(f"s_loss {s_loss:.10f}\t s_align: {s_align:.10f}\t s_unif {s_unif:.10f}\t")
  print(f"p_loss {p_loss:.10f}\t p_align: {p_align:.10f}\t p_unif {p_unif:.10f}\t")
  print("")

print("stability test")
for eps in jnp.linspace(1.0, 0.001, 5):
  temp = 0.5
  key, subkey = random.split(key)
  encodings = random.uniform(subkey, (jax.device_count(), batch_size, encoding_dim), dtype=jnp.float32)
  encod_a, encod_b = jnp.array_split(jnp.reshape(encodings, (-1, encodings.shape[-1])), 2)
 
  encod_a, encod_b = torch.tensor(np.array(encod_a)), torch.tensor(np.array(encod_b))
  pytorch_ntxent = torch_loss.NTXEntLoss(temperature = temp)
  t_loss = pytorch_ntxent(encod_a, encod_b)
  t_loss = t_loss.item()
  (t_align, t_unif) = (pytorch_ntxent.alignment.item(), pytorch_ntxent.uniformity.item() * temp)

  s_loss, (s_align, s_unif) = objective.pytorch_ported_ntxent(encodings, temp = temp)
  p_loss, (p_align, p_unif) = p_ntxent(encodings, temp = temp)

  print(f"eps: {eps}")
  print(f"temp: {temp}")
  print(f"t_loss {t_loss:.10f}\t t_align: {t_align:.10f}\t t_unif {t_unif:.10f}\t")
  print(f"s_loss {s_loss:.10f}\t s_align: {s_align:.10f}\t s_unif {s_unif:.10f}\t")
  print(f"p_loss {p_loss:.10f}\t p_align: {p_align:.10f}\t p_unif {p_unif:.10f}\t")
  print("")

# %%

# import optax._src.numerics as numerics

# encodings = random.uniform(subkey, (batch_size, encoding_dim), dtype=jnp.float32)


# lnalg = jnp.linalg.norm(encodings, ord = 2, axis = -1, keepdims=True)


# jax.vmap(functools.partial(numerics.safe_norm, min_norm = 1e-5))(encodings)