##
import logging
from typing import Any, Callable, Tuple

from ml_collections.config_dict.config_dict import create
from data import TrainIterator
from absl import logging

import data
import init, models, objective as obj

import jax, optax

# jax imports
from jax import random, numpy as jnp, tree_util
import jax.scipy.optimize as opt
from jax.numpy import ndarray

import numpy as np

from ml_collections import ConfigDict

import flax
from flax.training import train_state, checkpoints as chkp, common_utils
from flax import jax_utils, struct, linen
import functools
from tensorflow_probability.substrates import jax as tfp


class EncoderState(struct.PyTreeNode):
    apply_fn: Callable = struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any]

    def apply(self, x):
        return self.apply_fn(self.params, x, train=False, mutable=False)

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)


def create_and_initialise(
    key: ndarray, input_dim: int, num_classes: int, num_heads: int = 1
):
    logging.info(f"Initialising {num_heads} classification heads")
    clf = linen.Dense(features=num_classes * num_heads)
    params = clf.init(key, jnp.ones((0, input_dim)))
    return clf.apply, params


def concat_params(params):
    flat_params, params_tree = jax.tree_flatten(params)
    params_shape = [x.shape for x in flat_params]
    flat_params = jnp.concatenate([x.reshape(-1) for x in flat_params])
    return flat_params, (params_shape, params_tree)


def unconcat_params(param_vector, params_shape, params_tree):
    params_splits = np.cumsum([np.prod(s) for s in params_shape[:-1]])
    split_params = jnp.split(param_vector, params_splits)
    flat_params = [x.reshape(s) for x, s in zip(split_params, params_shape)]
    params = jax.tree_unflatten(params_tree, flat_params)
    return params


@jax.pmap
def encode(encod_state: EncoderState, batch_images: ndarray):
    return encod_state.apply_fn(
        encod_state.params, batch_images, train=False, mutable=False
    )


def process_dataset(
    encod_state: EncoderState, dataset_iter: TrainIterator, callback: Any = None
):
    encod_state = jax_utils.replicate(encod_state)
    encodings = []
    labels = []
    callbacks = []

    logging.info("Processing dataset")
    for batch_images, batch_labels in dataset_iter:
        batch_encodings = encode(encod_state, batch_images)

        if callback is None:
            encodings.append(batch_encodings.reshape(-1, batch_encodings.shape[-1]))
            labels.append(batch_labels.reshape(-1, batch_labels.shape[-1]))
        else:
            callbacks.append(callback(batch_encodings, batch_labels))

    if callback is None:
        encodings = jnp.concatenate(encodings)
        labels = jnp.concatenate(labels)
        return encodings, labels
    else:
        return callbacks


v_softmax_cross_entropy = jax.vmap(optax.softmax_cross_entropy)
v_accuracy = jax.vmap(obj.accuracy)


def evaluate_dataset(
    encod_state: EncoderState,
    clf_apply: Callable,
    params,
    test_iter: TrainIterator,
    num_heads: int,
):
    @jax.pmap
    def compute_accuracy(encodings, labels, params):
        logits = clf_apply(params, encodings)
        logits = logits.reshape(num_heads, -1, labels.shape[-1])
        labels = jnp.tile(labels, (num_heads, 1, 1))
        return v_accuracy(logits=logits, labels=labels).mean(-1)

    _compute_accuracy = functools.partial(
        compute_accuracy, params=jax_utils.replicate(params)
    )

    logging.info("Evaluating accuracy")
    accuracy_by_batch = process_dataset(
        encod_state, test_iter, callback=_compute_accuracy
    )

    accuracy_by_head = common_utils.get_metrics(accuracy_by_batch).mean(0)

    return accuracy_by_head


def train_and_evaluate(
    key: ndarray,
    encod_state: EncoderState,
    train_iter: TrainIterator,
    test_iter: TrainIterator,
    *,
    l2coeff_min=0,
    l2coeff_max=0.5,
    num_heads: int = 1,
    **lbfgs_kwargs,
) -> ndarray:
    logging.info("Starting linear evluation")
    encodings, labels = process_dataset(encod_state, train_iter)
    input_dim = encodings.shape[-1]
    num_classes = labels.shape[-1]

    clf_apply, params = create_and_initialise(key, input_dim, num_classes, num_heads)
    initial_params, shape_tree = concat_params(params)

    @functools.partial(jax.jit, static_argnums=[1])
    def loss_fn(
        params: Any,
        num_heads: int,
        l2coeffs: ndarray,
        encodings: ndarray,
        labels: ndarray,
    ):
        logits = clf_apply(params, encodings)

        logits_by_head = logits.reshape(num_heads, -1, labels.shape[-1])
        labels_by_head = jnp.tile(labels, (num_heads, 1, 1))

        loss = v_softmax_cross_entropy(logits_by_head, labels_by_head).sum(0).mean()

        l2norms = jax.tree_map(lambda x: jnp.sum(x ** 2), jax.tree_flatten(params)[0])
        reg = sum([norm * coeff for norm, coeff in zip(l2norms, l2coeffs)])

        return loss + reg

    # repeat twice for bias & weight layers
    l2coeffs = jnp.repeat(jnp.linspace(l2coeff_min, l2coeff_max, num_heads), 2)

    @jax.value_and_grad
    def _loss_value_and_grad(params_vector):
        params = unconcat_params(params_vector, *shape_tree)
        return loss_fn(params, num_heads, l2coeffs, encodings, labels)

    loss_value_and_grad = jax.jit(_loss_value_and_grad)

    logging.info(f"Training classifiers")
    result = tfp.optimizer.lbfgs_minimize(
        loss_value_and_grad, initial_position=initial_params, **lbfgs_kwargs
    )

    print(result)

    params = unconcat_params(result.position, *shape_tree)
    accuracy_by_head = evaluate_dataset(
        encod_state, clf_apply, params, test_iter, num_heads
    )
    return accuracy_by_head


##

import defaults, init, os

config = defaults.get_config()
train_iter = data.create_input_iter(config.clf_config, dataset="cifar10")
train_iter.set_epochs(1)
test_iter = data.create_input_iter(config.clf_config, dataset="cifar10", split="test")
test_iter.set_epochs(1)
key = random.PRNGKey(0)
num_heads = 3
num_classes = 10
l2coeff = 1e-3
##

encod_state = init.restore_encoder_state(
    config, os.path.join("checkpoints", "morning-river-67"), image_shape=(32, 32, 3)
)

##
logging.set_verbosity(logging.INFO)
acc = train_and_evaluate(
    key,
    encod_state,
    train_iter,
    test_iter,
    num_heads=num_heads,
    max_iterations=500,
)
