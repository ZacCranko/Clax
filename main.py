import train
import defaults
from absl import logging
#%%
import wandb
import jax 

if __name__ == "__main__":
  assert jax.local_device_count() > 1

  wandb.init(project='jax', entity='zaccranko')
  config = defaults.get_config()
  config.projector = "CIFAR10Classifier"
  logging.set_verbosity(logging.INFO)
  print(config)
  train.train_and_evaluate(config)