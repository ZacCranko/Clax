import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from absl import app, flags, logging
import jax, train, defaults
from ml_collections.config_flags import config_flags
from datetime import datetime

import wandb
wandb.init(sync_tensorboard=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_dict('config', defaults.get_config())

logging.set_verbosity(logging.INFO)

def get_workdir(*, logdir: str, name: str) -> str:
  time_string = datetime.utcnow().replace(microsecond=0).isoformat().replace(":", "_")
  return os.path.join(logdir, name)

def main(argv):
  wandb.config.update(FLAGS.config.to_dict())

  global key 
  key = jax.random.PRNGKey(0)

  config = FLAGS.config
  workdir = get_workdir(logdir = "tensorboard", name = wandb.run.name)

  train.train_and_evaluate(key, config, workdir = workdir)

if __name__ == '__main__':
  if jax.local_device_count() % 2 != 0:
    raise RuntimeError("An even number of XLA devices is required.")
  app.run(main)