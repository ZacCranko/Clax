import logging
from typing import Any, Callable, Tuple
from jax.numpy import ndarray
from init import CLTrainState

from data import TrainIterator
from absl import logging

import init
import jax
from jax import numpy as jnp
import numpy as np
from flax import jax_utils
from tqdm import tqdm
from . import linear_eval


@jax.pmap
def encode(state_repl: CLTrainState, images: ndarray):
  params = dict(params=state_repl.params["backbone"],
                batch_stats=state_repl.batch_stats["backbone"])
  return state_repl.backbone_apply_fn(params,
                                      images,
                                      train=False,
                                      mutable=False)


p_concatenate = jax.pmap(jnp.concatenate)


def process_dataset(state_repl: init.CLTrainState, dataset_iter: TrainIterator):
  encodings = []
  labels = []

  for batch_images, batch_labels in tqdm(dataset_iter, desc="Encoding"):
    batch_encodings = encode(state_repl, batch_images)
    encodings.append(batch_encodings.reshape(-1, batch_encodings.shape[-1]))
    labels.append(batch_labels.reshape(-1, batch_labels.shape[-1]))

  return jnp.concatenate(encodings), jnp.concatenate(labels)


def linear_accuracy(state_repl: CLTrainState, train_iter, val_iter):
  train_encodings, train_labels = process_dataset(state_repl, train_iter)
  # val_encodings, val_labels = process_dataset(state_repl, val_iter)

  ((distributed_train_encodings, distributed_train_labels),
   distributed_train_mask) = linear_eval.reshape_and_pad_data_for_devices(
       (train_encodings, np.argmax(train_labels, axis=1)))

  logging.info("Training linear_classifiers")
  weights, biases, res = linear_eval.train(distributed_train_encodings,
                                           distributed_train_labels,
                                           distributed_train_mask,
                                           l2_regularization=1e-6)

  accuracy = linear_eval.evaluate(distributed_train_encodings,
                                  distributed_train_labels,
                                  distributed_train_mask, weights, biases)

  logging.info(
      f"Linear accuracy: {accuracy:.2%} at step {jax_utils.unreplicate(state_repl.step)}"
  )

  return accuracy
