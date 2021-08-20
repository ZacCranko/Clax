import ml_collections


def get_clf_config(batch_size: int = 1024):
    """Get the default hyperparameter configuration."""
    clf_config = ml_collections.ConfigDict()

    clf_config.learning_rate = 1.0
    clf_config.momentum = 0.9
    clf_config.num_epochs = 15
    clf_config.batch_size = batch_size
    clf_config.cache_dataset = True
    clf_config.l2coeff = 0.001

    return clf_config

def get_config():
    config = ml_collections.ConfigDict()
    
    # `name` argument of tensorflow_datasets.builder()
    # config.dataset = 'imagenet2012:5.*.*'

    config.dataset = "cifar10"

    # As defined in the `models` module.
    config.stem = "CIFAR"
    config.model = "ResNet50"
    config.projector = "SimCLR"

    config.learning_rate = 1.0
    config.warmup_epochs = 10
    config.momentum = 0.9
    config.batch_size = 2048
    config.step = 0
    config.num_epochs = 500

    config.cache_dataset = True
    config.half_precision = False

    config.clf_config = get_clf_config(batch_size = 2 * config.batch_size)
    
    config.linear_eval_step_freq = 200

    config.num_train_steps = -1
    config.steps_per_eval = -1
    config.log_every_steps = 1

    return config
