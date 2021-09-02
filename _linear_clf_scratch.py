##
from typing import Any, Callable, Tuple

from ml_collections.config_dict.config_dict import create
from data import TrainIterator
from absl import logging

import data
import init, models, objective as obj

import jax, optax
from jax import random, numpy as jnp, tree_util
from jax.numpy import ndarray

import numpy as np

from ml_collections import ConfigDict

import flax
from flax.training import train_state, checkpoints as chkp, common_utils
from flax import jax_utils, struct, linen
import functools


class EncoderState(struct.PyTreeNode):
    apply_fn: Callable = struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any]

    def apply(self, x, train: bool = True):
        return self.apply_fn(self.params, x, train=train, mutable=False)

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)


def create_and_initialise(
    key: ndarray, input_dim: int, num_classes: int, num_heads: int = 1
):
    clf = linen.Dense(features=num_classes * num_heads)
    params = clf.init(key, jnp.ones((0, input_dim)))
    return clf.apply, params


def concat_params(params):
    flat_params, params_tree = jax.tree_flatten(params)
    params_shape = [x.shape for x in flat_params]
    params_splits = np.cumsum([np.prod(s) for s in params_shape[:-1]])
    return (
        jnp.concatenate([x.reshape(-1) for x in flat_params]),
        (
            params_splits,
            params_tree,
        ),
    )


##
import defaults, init

config = defaults.get_config()
encod_state = init.restore_encoder_state(
    config, os.path.join("checkpoints", "morning-river-67"), image_shape=(32, 32, 3)
)

##
@jax.pmap
def cross_entropy(params, images, labels):
    return optax.softmax_cross_entropy(apply(params, images), labels)


@jax.pmap
def forward(params, batch):
    return


key = random.PRNGKey(0)
num_heads = 2
num_classes = 10
input_dim = 2048
apply, params = create_and_initialise(key, input_dim, num_classes, num_heads)


dummy = random.normal(key, (8 * 128, 2048))

jax.device_put_replicated(params, jax.local_devices())

##


def loss_fn(params, l2coeff: float, images, labels):
    softmax = optax.softmax_cross_entropy(apply(params, images), labels)
    leaves, tree_def = jax.tree_flatten(params)
    reg = sum(jax.tree_map(lambda x: jnp.sum(x ** 2), leaves))
    return softmax + l2coeff * reg


@jax.value_and_grad
def objective(param_vector):
    split_params = jnp.split(
        param_vector, np.cumsum([np.prod(s) for s in params_shape[:-1]])
    )
    flat_params = [x.reshape(s) for x, s in zip(split_params, params_shape)]
    params = jax.tree_unflatten(params_tree, flat_params)

    softmax = optax.softmax_cross_entropy(apply(params, images), labels)
    leaves, tree_def = jax.tree_flatten(params)
    reg = sum(jax.tree_map(lambda x: jnp.sum(x ** 2), leaves))

    loss = softmax + l2coeff * reg

    return loss_fn(params, *batch)


##

apply, params = create_and_initialise(key, input_dim, num_classes, num_heads)

initial_point, (params_tree, params_shape) = concat_params(params)


# num_heasd
# def loss_params(leaf, num_heads):
#     (leaf ** 2).reshape(num_heads, leaf.shape[0], -1)

# map_fn = functools.partial(loss_params, 1)
# jax.tree_map(map_fn, params)

num_heads = 1
num_classes = 10
input_dim = 2048

key = random.PRNGKey(0)
apply, params = create_and_initialise(key, input_dim, num_classes, num_heads)

samples = random.normal(key, (128, 2048))

bias = params["params"]["bias"]
kernel = params["params"]["kernel"]


leaves, tree_def = jax.tree_flatten(params)
sum(jax.tree_map(lambda x: jnp.sum(x ** 2), leaves))
