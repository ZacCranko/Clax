##
import logging
from typing import Any, Callable, Tuple

from data import TrainIterator
from absl import logging

import init, models, objective as obj

import jax, optax

# jax imports
from jax import numpy as jnp
import jax.scipy.optimize as opt 
from jax.numpy import ndarray

import numpy as np

import flax
from flax.training import common_utils
from flax import jax_utils, linen
from functools import partial
from tensorflow_probability.substrates import jax as tfp
from jax.interpreters import xla

from tqdm import tqdm

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
def encode(state_repl: init.CLTrainState, images: ndarray):
    return state_repl.encode(images)

def process_dataset(
    state_repl: init.CLTrainState, dataset_iter: TrainIterator, callback: Any = None
):
    encodings = []
    labels = []
    callbacks = []

    logging.info("Processing dataset")

    for batch_images, batch_labels in tqdm(dataset_iter):
        batch_encodings = encode(state_repl, batch_images)

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
    state_repl: init.CLTrainState,
    test_iter: TrainIterator,
    num_heads: int,
):
    @jax.pmap
    def compute_accuracy(encodings, labels, state):
        logits = state.classify(encodings)
        labels = jnp.tile(labels, (num_heads, 1, 1))
        return v_accuracy(logits=logits, labels=labels).mean(-1)

    _compute_accuracy = partial(compute_accuracy, state=state_repl)

    logging.info("Evaluating accuracy")
    accuracy_by_batch = process_dataset(
        state_repl, test_iter, callback=_compute_accuracy
    )

    accuracy_by_head = common_utils.get_metrics(accuracy_by_batch).mean(0)

    return accuracy_by_head

def train_and_evaluate(
    key: ndarray,
    state_repl: init.CLTrainState,
    train_iter: TrainIterator,
    test_iter: TrainIterator,
    *,
    minl2coeff=0,
    maxl2coeff=0.05,
    num_heads: int = 10,
    max_iterations=500,
    tolerance=1e-3,
    **lbfgs_kwargs,
) -> Tuple[init.CLTrainState, ndarray]:
    logging.info(
        f"Starting linear evluation using {num_heads} weights in [{minl2coeff}, {maxl2coeff}]"
    )

    encodings, labels = process_dataset(state_repl, train_iter)
    l2coeffs = jnp.linspace(minl2coeff, maxl2coeff, num_heads)
    
    state = jax_utils.unreplicate(state_repl)
    initial_params, shape_tree = concat_params(state.clf_heads_params)

    def loss_fn(
        params: Any,
        l2coeffs: ndarray,
        encodings: ndarray,
        labels: ndarray,
        state: init.CLTrainState,
    ):
        param_leaves = jax.tree_leaves(params)

        num_heads = len(param_leaves) // 2

        logits = state.clf_heads_apply_fn(dict(params=params), encodings)
        labels = jnp.tile(labels, (num_heads, 1, 1))
        loss = v_softmax_cross_entropy(logits, labels).mean()

        # repeat for weight and bias leaves
        l2coeffs = jnp.repeat(l2coeffs, 2)
        reg = (
            sum([jnp.sum(w ** 2) * coeff for w, coeff in zip(param_leaves, l2coeffs)])
            / num_heads
        )
        return loss + reg

    @jax.value_and_grad
    def loss_value_and_grad(params_vector, l2coeffs, encodings, labels, state):
        params = unconcat_params(params_vector, *shape_tree)
        return loss_fn(params, l2coeffs, encodings, labels, state)
    
    obj = partial(
        loss_value_and_grad,
        l2coeffs=l2coeffs,
        encodings=encodings,
        labels=labels,
        state=state
    )

    
    logging.info(f"Training classifiers")

    # jax.scipy BFGS code, (no L-)
    # args = (l2coeffs, encodings, labels, state)
    # result = opt.minimize(loss_fn, initial_params, args, method = "BFGS", options = dict(disp = True))

    result = tfp.optimizer.lbfgs_minimize(
        jax.jit(obj),
        initial_position=initial_params,
        max_iterations=max_iterations,
        tolerance=tolerance,
        **lbfgs_kwargs,
    )
    if result.converged:
        logging.info(f"L-BFGS converged in {result.num_iterations} steps")
    else:
        logging.warning(f"L-BFGS failed to converge in {result.num_iterations} steps")

    params = unconcat_params(result.position, *shape_tree)
    new_state = state.replace(clf_heads_params = params)

    new_state_repl = jax_utils.replicate(new_state)
    accuracy_by_head = evaluate_dataset(
        new_state_repl, test_iter, num_heads
    )

    max_head = jnp.argmax(accuracy_by_head)
    max_accuracy = accuracy_by_head[max_head]
    logging.info(
        f"Found max test accuracy {max_accuracy:.2%} at head {max_head + 1} of {num_heads}"
    )

    xla._xla_callable.cache_clear()

    return new_state_repl, max_accuracy