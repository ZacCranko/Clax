from absl.testing import absltest
import sys
sys.path.append("..")
import main, defaults, loss_functions

import jax
from jax import random, numpy as jnp
from parameterized import parameterized


class ParallelNTXEntCorrectnessTest(absltest.TestCase):
  @parameterized.expand([
    [1.0], [0.5], [0.25], [0.1]
  ])
  def test_parallel_ntxent_correctness(self, temp):
    batch_size = 128
    encoding_dim = 2048
    encodings = random.uniform(random.PRNGKey(20), (jax.device_count(), batch_size, encoding_dim), dtype=jnp.bfloat16)

    s_loss, (s_align, s_unif) = loss_functions.pytorch_ported_ntxent(encodings, temp = temp)
    p_loss, (p_align, p_unif) = loss_functions.p_ntxent(encodings, temp = temp)

    self.assertAlmostEqual(s_loss, p_loss)
    self.assertAlmostEqual(s_align, p_align)
    self.assertAlmostEqual(s_unif, p_unif)


class ResNetTest(absltest.TestCase):
  @parameterized.expand([
    ["_ResNet1", "CIFAR", 32, 8, 64],
    ["_ResNet1", "ImageNet", 224, 8, 64],
    ["ResNet50", "ImageNet", 224, 8, 2048]
  ])
  def test_create_model(self, resnet, stem, image_size, batch_size, out_shape):
    model = main.create_model(resnet = resnet, stem = stem, half_precision=False)
    params, batch_stats = main.initialized(random.PRNGKey(1), image_size, model)
    variables = {'params': params, 'batch_stats': batch_stats}

    inp = random.normal(random.PRNGKey(image_size), (batch_size, image_size, image_size, 3))
    out = model.apply(variables, inp, train = False)

    self.assertEqual(out.shape, (batch_size, out_shape))

class TrainStateTest(absltest.TestCase):
  def test_create_train_state(self):
    config = defaults.get_config()
    config.warmup_epochs = 10
    config.num_epochs = 1000
    config.dtype = jnp.float16
    learning_rate_fn = main.create_learning_rate_fn(config, 1.5, 100)

    model = main.create_model(resnet = "_ResNet1")
    image_size = 32
    state = main.create_train_state(random.PRNGKey(0), config, model, image_size, learning_rate_fn)
    self.assertIsNotNone(state)


if __name__ == '__main__':
  absltest.main()