import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import train
import defaults
from absl import logging
#%%
import wandb
import jax 

if __name__ == "__main__":
  if jax.local_device_count() % 2 != 0:
    raise RuntimeError("An even number of XLA devices is required.")

  wandb.init(project='jax', entity='zaccranko')
  config = defaults.get_config()
  logging.set_verbosity(logging.INFO)
  
  print(config)
  state = train.train_and_evaluate(config)