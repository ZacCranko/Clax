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

from sklearn import model_selection


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


def linear_accuracy(state_repl: CLTrainState,
                    data_iterator: TrainIterator,
                    test_size: float = 0.25):
  encodings, labels = process_dataset(state_repl, data_iterator)

  labels = np.argmax(labels, axis=1)

  (train_encodings, test_encodings, train_labels,
   test_labels) = model_selection.train_test_split(encodings,
                                                   labels,
                                                   test_size=test_size)

  ((distributed_train_encodings, distributed_train_labels),
   distributed_train_mask) = linear_eval.reshape_and_pad_data_for_devices(
       (train_encodings, train_labels))

  ((distributed_test_encodings, distributed_test_labels),
   distributed_test_mask) = linear_eval.reshape_and_pad_data_for_devices(
       (test_encodings, test_labels))

  logging.info("Training linear_classifiers")
  weights, biases, res = linear_eval.train(distributed_train_encodings,
                                           distributed_train_labels,
                                           distributed_train_mask,
                                           l2_regularization=1e-6)

  accuracy = linear_eval.evaluate(distributed_test_encodings,
                                  distributed_test_labels,
                                  distributed_test_mask, weights, biases)

  logging.info(
      f"Linear accuracy: {accuracy:.2%} at step {jax_utils.unreplicate(state_repl.step)}"
  )

  return accuracy
