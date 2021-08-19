import ml_collections

def get_clf_config():
    """Get the default hyperparameter configuration."""
    clf_config = ml_collections.ConfigDict()

    clf_config.learning_rate = 1.0
    clf_config.momentum = 0.9
    clf_config.num_epochs = 5
    clf_config.batch_size = 2048
    clf_config.cache_dataset = True
    clf_config.l2coeff = 0.001

    return clf_config

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # As defined in the `models` module.
    config.stem = "CIFAR"
    config.model = "ResNet50"
    config.projector = "CIFAR10Classifier"

    # `name` argument of tensorflow_datasets.builder()
    # config.dataset = 'imagenet2012:5.*.*'
    config.dataset = "cifar10"

    config.learning_rate = 1.0
    config.warmup_epochs = 10
    config.momentum = 0.9
    config.batch_size = 1024
  
    config.num_epochs = 30
    config.log_every_steps = 1

    config.cache_dataset = True
    config.half_precision = False

    config.step = 0

    config.clf_config = get_clf_config()

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1
    return config
