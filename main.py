import train
import defaults
from absl import logging
#%%
import wandb
import jax 

if __name__ == "__main__":
  if jax.local_device_count() % 2 != 0:
    raise RuntimeError("Parallel execution requires an even number of XLA devices.")

  wandb.init(project='jax', entity='zaccranko')
  config = defaults.get_config()
  config.projector = "CIFAR10Classifier"
  logging.set_verbosity(logging.INFO)
  
  print(config)
  train.train_and_evaluate(config)