import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from absl import app, flags, logging
import jax, train, defaults
from datetime import datetime

import json
from ml_collections.config_flags import config_flags
from ml_collections import ConfigDict

import wandb

FLAGS = flags.FLAGS
config_flags.DEFINE_config_dict('config', defaults.get_config())

logging.set_verbosity(logging.INFO)


def save_config(config: ConfigDict, logdir: str):
  with open(os.path.join(logdir, "config.json"), 'w+') as f:
      json.dump(config.to_dict(), f, indent=4)

def get_workdir(*, logdir: str, name: str) -> str:
  time_string = datetime.utcnow().replace(microsecond=0).isoformat().replace(":", "_")
  return os.path.join(logdir, name)

def main(argv):
  if jax.process_index() == 0:
    wandb.init()
    wandb.config.update(FLAGS.config.to_dict())
    wandb.config.seed = datetime.now().microsecond
    workdir = get_workdir(logdir = "checkpoints", name = wandb.run.name)
    save_config(config, workdir)

    train.train_and_evaluate(key, config, workdir = workdir)


  global key 
  key = jax.random.PRNGKey(wandb.config.seed)
  config = FLAGS.config


if __name__ == '__main__':
  if jax.local_device_count() % 2 != 0:
    raise RuntimeError("An even number of XLA devices is required.")
  app.run(main)