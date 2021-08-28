from typing import Any, Callable, Tuple
from absl import logging

import data
import init, models, objective as obj

import jax, optax 
from jax import random, numpy as jnp, tree_util 

from ml_collections import ConfigDict

import flax 
from flax.training import train_state, checkpoints as chkp, common_utils
from flax import jax_utils, struct
import functools

class EncoderState(struct.PyTreeNode):
  apply_fn: Callable = struct.field(pytree_node=False)
  params: flax.core.FrozenDict[str, Any]

  def apply(self, x, train: bool = True):
    return self.apply_fn(self.params, x, train = train, mutable = False)

  @classmethod
  def create(cls, **kwargs):
    return cls(**kwargs)

def initialized(rng, num_classes: int, num_features: int, num_heads: int):
  clf = models.classifier.MutiHeadClassifier(num_heads = num_heads, num_classes = num_classes)
  
  @jax.jit
  def init(*args):
      return clf.init(*args)

  params = init(rng, jnp.ones((128, num_features)))
  return clf, params

def create_train_state(rng: random.PRNGKey, 
                       clf_config: ConfigDict, 
                       state, assembly, image_shape, 
                       num_classes: int, num_heads: int):
  
  params = flax.core.frozen_dict.freeze({"params" : state.params["backbone"], "batch_stats" : state.batch_stats["backbone"]})
  encod_state = EncoderState.create(apply_fn = assembly.backbone.apply, params = params)
  _, num_features = encod_state.apply(jnp.ones((128,) + image_shape), train = False).shape
  
  # initialise model
  clf, params = initialized(rng, num_classes, num_features, num_heads)
  tx = optax.adam(learning_rate=clf_config.learning_rate)

  clf_state = train_state.TrainState.create(apply_fn=clf.apply, params=params, tx=tx)
  return encod_state, clf_state

v_softmax_cross_entropy = jax.vmap(optax.softmax_cross_entropy)
v_accuracy = jax.vmap(obj.accuracy)

@functools.partial(jax.pmap, axis_name = "batch")
def train_step(rep_encod_state: EncoderState, 
               clf_state: train_state.TrainState,
               l2coeffs,
               images, labels):

  num_heads = len(clf_state.params['params'])
  labels_by_head = jnp.stack([labels for _ in range(num_heads)], axis = 0)
  encodings = rep_encod_state.apply(images, train = False)

  def loss_fn(params):
    logits_by_head = clf_state.apply_fn(params, encodings)

    cross_entropy = v_softmax_cross_entropy(logits_by_head, labels_by_head).mean(-1).sum()
    acc =  v_accuracy(logits = logits_by_head, labels = labels_by_head).mean()
    
    metrics = {'cross_entropy' : cross_entropy, 'accuracy' : acc}

    params = jax.tree_leaves(clf_state.params['params'])
    sum_l2_reg = sum([
      jnp.sum(w ** 2) * coeff 
      for w, coeff in zip(params, jnp.repeat(l2coeffs, 2))])
    
    metrics['l2reg'] = sum_l2_reg

    loss = cross_entropy + sum_l2_reg

    return loss, metrics

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  # eval gradients 
  (_, metrics), grads = grad_fn(clf_state.params)
  
  grads = jax.lax.pmean(grads, axis_name = "batch")
  new_clf_state = clf_state.apply_gradients(grads = grads)

  return new_clf_state, jax.lax.pmean(metrics, axis_name = "batch")

@functools.partial(jax.pmap, axis_name = "batch")
def multi_head_accuracy(encod_state, rep_clf_state, images, labels):
    encodings = encod_state.apply(images, train = False)
    logits_by_head = rep_clf_state.apply_fn(rep_clf_state.params, encodings)
    num_heads = len(rep_clf_state.params['params'])
    labels_by_head = jnp.stack([labels for _ in range(num_heads)], axis = 0)
    
    acc = v_accuracy(logits = logits_by_head, labels = labels_by_head)

    metrics = {
      "accuracy" : acc
    }

    return jax.lax.pmean(metrics, axis_name = "batch")

def compute_metrics(rep_encod_state: train_state.TrainState, 
                    rep_clf_state: train_state.TrainState, 
                    data_iter: data.TrainIterator):

  metrics_by_batch = []

  logging.info("Computing eval metrics")
  for batch in data_iter:
    metrics = multi_head_accuracy(rep_encod_state, rep_clf_state, *batch)
    metrics_by_batch.append(metrics)

  get_metrics = common_utils.get_metrics(metrics_by_batch)
  metrics_by_head = jax.tree_map(lambda x: jnp.mean(x, 0), get_metrics)

  return metrics_by_head

def train_and_evaluate(encod_state: train_state.TrainState, clf_state: train_state.TrainState, 
                       train_iter: data.TrainIterator, test_iter: data.TrainIterator, l2coeffs):
  
  encod_state = jax_utils.replicate(encod_state)
  clf_state = jax_utils.replicate(clf_state)
  l2coeffs = jax_utils.replicate(l2coeffs)

  for batch in train_iter: 
    clf_state, metrics = train_step(encod_state, clf_state, l2coeffs, *batch)

    # if train_iter.is_epoch_end():
      # metrics_str = " ".join(f"{metric}: {val:.2e}" for metric,val in jax_utils.unreplicate(metrics).items())
      # logging.info(f"epoch {train_iter.get_epoch()} {metrics_str}")
      
  test_metrics = compute_metrics(encod_state, clf_state, test_iter)

  return test_metrics, jax_utils.unreplicate(clf_state)

def linear_accuracy(key: random.PRNGKey, config: ConfigDict, state: init.CLTrainState, 
                    assembly: flax.linen.Module, train_iter: data.TrainIterator, test_iter: data.TrainIterator,
                    minl2coeff: float = 0, maxl2coeff: float = 0.01, num_heads: int = 10):
  subkey, key = random.split(key)

  encod_state, clf_state = create_train_state(subkey, config.clf_config, state, assembly,
                              train_iter.image_shape, train_iter.num_classes, num_heads)
  

  l2coeffs = jnp.linspace(minl2coeff, maxl2coeff, num_heads)
  
  logging.info("Training classifier heads")
  test_metrics, clf_state = train_and_evaluate(encod_state, clf_state, train_iter, test_iter, l2coeffs)

  logging.info("Linear training finished")
  for head, (coeff, acc) in enumerate(zip(l2coeffs, test_metrics["accuracy"])):
    logging.info(f"head_{head}: l2coeff: {coeff:.2e}, accuracy: {acc:.2%}")

  accuracy = jnp.max(test_metrics['accuracy'])
  
  return accuracy, test_metrics

