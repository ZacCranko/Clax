## Types
from typing import Any, Callable, Tuple
from concurrent.futures import process
from ml_collections import ConfigDict
from init import CLTrainState
from data import TrainIterator
from jax.numpy import ndarray

from absl import logging
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


from functools import partial
import jax
from jax import numpy as jnp

import objective as obj

import numpy as np

@jax.pmap
def encode(state_repl: CLTrainState, images: ndarray):
    return state_repl.encode(images)

# concatenate arrays on each xla device before transferring back to host
p_concatenate = jax.pmap(jnp.concatenate)

def process_dataset(
    state_repl: CLTrainState, dataset_iter: TrainIterator, callback: Any = None
):
    encodings = []
    labels = []
    callbacks = []

    logging.info("Processing dataset")

    for batch_images, batch_labels in tqdm(dataset_iter):
        batch_encodings = encode(state_repl, batch_images)

        if callback is None:
            encodings.append(batch_encodings)
            labels.append(batch_labels)
        else:
            callbacks.append(callback(batch_encodings, batch_labels))

    if callback is None:
        encodings = p_concatenate(encodings)
        labels = p_concatenate(labels)
        
        encodings = encodings.reshape(-1, encodings.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])

        return encodings, labels
    else:
        return callbacks

v_accuracy = jax.vmap(obj.accuracy)

def linear_clf(*,input_dim: int = 128, num_classes: int = 10, l2coeff: float = 0.01):
    clf = Sequential(Dense(num_classes, activation='softmax', input_dim = input_dim, kernel_regularizer=l1_l2(l2=l2coeff)))
    clf.compile(loss='categorical_crossentropy', optimizer="adagrad", metrics=["accuracy"])
    return clf 

def linear_accuracy(
    config: ConfigDict,
    state_repl: CLTrainState,
    train_iter: TrainIterator,
    minl2coeff: float = 0,
    maxl2coeff: float = 0.01,
):
    # jnp.asarray casues issues with clf.fit below so we have to copy to np arrays, I don't know why
    encodings, labels = process_dataset(state_repl, train_iter)
    encodings = np.array(encodings, copy=False)
    labels = np.array(labels, copy=False)
    
    input_dim, num_classes = encodings.shape[-1], labels.shape[-1]

    linear_clf_init = partial(linear_clf, input_dim = input_dim, num_classes = num_classes, l2coeff = 0.001)
    clf = KerasClassifier(linear_clf_init, epochs=50, batch_size=1024, verbose=0)
    
    fit_result = clf.fit(encodings, labels)
    accuracy = fit_result.history['accuracy'][-1]

    del encodings, labels 
    tf.keras.backend.clear_session()

    logging.info(f"Linear training finished with {accuracy:.2%} accuracy")
    
    return accuracy