import logging
from typing import Any, Callable, Tuple, List
from jax.numpy import ndarray
from init import CLTrainState

from data import TrainIterator
from absl import logging

import functools
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


concatenate = jax.pmap(jnp.concatenate)


def concatenate_and_collect(batch_list: List[ndarray]) -> ndarray:
  """Converts a list of batches sharded on d devices into a continuous array.

  Args:
      batch_list (List[ndarray]): list of arrays of batches with shape (d x *(a x b x c ...)).

  Returns:
      ndarray: Continuous array of shape (a * d * n x b x c ...), where n is the length of the list.
  """
  batch = concatenate(batch_list)
  return batch.reshape(-1, *batch.shape[2:])


def process_dataset(state_repl: init.CLTrainState, dataset_iter: TrainIterator):
  encodings = []
  labels = []

  for batch_images, batch_labels in tqdm(dataset_iter, desc="Encoding"):
    batch_encodings = encode(state_repl, batch_images)

    encodings.append(batch_encodings)
    labels.append(batch_labels)

  return concatenate_and_collect(encodings), concatenate_and_collect(labels)


def linear_accuracy(state_repl: CLTrainState,
                    data_iterator: TrainIterator,
                    test_size: float = 0.25) -> float:
  """Compute the linear accuracy of an encoder

  Args:
      state_repl (CLTrainState): CLTrainState replicated to each XLA device.
      data_iterator (TrainIterator): Iterator from which to generate the encodings.
      test_size (float, optional): Fraction of the data_iterator to use for test evaluation. Defaults to 0.25.

  Returns:
      float: Linear accuracy as a float in [0,1]
  """
  encodings, labels = process_dataset(state_repl, data_iterator)

  # linear_eval.py expects numerical labels
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

  logging.info("Training linear classifiers")
  weights, biases, optimization_result = linear_eval.train(
      distributed_train_encodings,
      distributed_train_labels,
      distributed_train_mask,
      l2_regularization=1e-5,
      tolerance=1e-4,
  )

  accuracy = linear_eval.evaluate(distributed_test_encodings,
                                  distributed_test_labels,
                                  distributed_test_mask, weights, biases)

  logging.info(
      f"Linear accuracy: {accuracy:.2%} at step {jax_utils.unreplicate(state_repl.step)}"
  )

  return accuracy
