##
import serialization as serial
from ml_collections.config_dict.config_dict import create
from tensorflow_datasets.core.naming import parse_builder_name_kwargs
import data, defaults, matplotlib.pyplot as plt 

import jax, ml_collections, optax 
from jax import random, numpy as jnp, tree_util 
from absl import logging 

from flax import jax_utils, struct

import linear_evaluation as eval
from absl import logging 
logging.set_verbosity(logging.INFO)
import objective as obj, optax, linear_evaluation as lineval
import importlib

config = defaults.get_config()
config.clf_config.num_epochs = 20
## 

train_iter  = data.create_input_iter(config.clf_config, is_training=False, split = 'train', dataset = config.dataset)
test_config = ml_collections.ConfigDict(config.clf_config.to_dict())
test_config.num_epochs = 1
test_iter  = data.create_input_iter(test_config, is_training=False, split = 'test', dataset = config.dataset)
_iter = iter(test_iter)
##
assembly, state = serial.restore_checkpoint(config, "tensorboard/visionary-pyramid-11", train_iter)
##
images, labels = next(iter(test_iter))
variables = {'params': state.params, 'batch_stats': state.batch_stats}
logits = assembly.apply(variables, images[0], train=False, mutable=False)
print(f"accuracy: {obj.accuracy(logits = logits, labels = labels[0]):.2%}")
print(f"softmax: {optax.softmax_cross_entropy(logits = logits, labels = labels[0]).mean():.2e}")

## 
importlib.reload(lineval)
config.clf_config.learning_rate = 1e-4
config.clf_config.num_epochs = 5
train_iter  = data.create_input_iter(config.clf_config, is_training=False, split = 'train', dataset = config.dataset)
acc = lineval.linear_accuracy(random.PRNGKey(0), config, state, assembly, train_iter, test_iter, minl2coeff=0, maxl2coeff=1, num_steps=4)
## 
from jax import tree_util

import jax.numpy as jnp
import flax
keys = clf_state.params['params'].keys()
leaves = flax.core.freeze({ head : l2coeff for head,l2coeff in zip(keys, jnp.linspace(0, 1, len(keys))) })


# ##
# rep_encod_state = jax_utils.replicate(encod_state)
# rep_clf_state = jax_utils.replicate(clf_state)
# l2coeffs = jax_utils.replicate(l2coeffs)


# state.apply(images[0])

# encod_state, clf_state = states
# encodings = encod_state.apply(images[0], train = False)
# variables = {'params': state.params['backbone'], 'batch_stats': state.batch_stats['backbone']}
# encodings = assembly.backbone.apply(variables, images[0], train=False, mutable=False)

# variables = {'params': state.params, 'batch_stats': state.batch_stats}
# assembly.apply(variables, images[0], train=False, mutable=False)


# logits_by_head = rep_clf_state.apply_fn({'params': state.params}, encodings, train=False)

# ##

##
