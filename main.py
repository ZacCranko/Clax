import os

from absl import app, flags, logging
import jax, train, defaults
from datetime import datetime

import json
from ml_collections.config_flags import config_flags
from ml_collections import ConfigDict

import wandb

# parse args
FLAGS = flags.FLAGS
config_flags.DEFINE_config_dict('config', defaults.get_config())

# attempt to suppress some tensorflow noise
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.INFO)


def save_config(config, logdir: str):
  if not os.path.exists(logdir):
    os.makedirs(logdir)

  with open(os.path.join(logdir, "config.json"), 'w+') as f:
    json.dump(dict(config), f, indent=4)


def main(argv):
  if jax.process_index() == 0:
    wandb.init(anonymous='allow')
    wandb.config.update(FLAGS.config.to_dict())
    wandb.config.seed = datetime.now().microsecond

  workdir = os.path.join("checkpoints", wandb.run.name)
  save_config(wandb.config, workdir)
  config = ConfigDict(dict(wandb.config))

  train.train_and_evaluate(config, workdir=workdir)


if __name__ == '__main__':
  if jax.local_device_count() % 2 != 0:
    xla_info = f"(got {jax.local_device_count()} XLA device(s): {', '.join(map(str, jax.local_devices()))})"
    raise RuntimeError(f"An even number of XLA devices is required {xla_info}")
  app.run(main)