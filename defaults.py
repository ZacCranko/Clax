import ml_collections

def get_clf_config(batch_size: int = 1024) -> ml_collections.ConfigDict:
    """Get the default hyperparameter configuration."""
    clf_config = ml_collections.ConfigDict()

    clf_config.learning_rate = 0.001
    clf_config.warmup_epochs = 2
    clf_config.momentum = 0.9
    clf_config.batch_size = 1024
    clf_config.cache_dataset = True

    clf_config.start_step = 0
    clf_config.num_epochs = 50
    clf_config.num_steps = -1

    # l2 regularizer coefficient
    clf_config.l2coeff = 0.001

    return clf_config

def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.clf_config = get_clf_config()

    # `name` argument of tensorflow_datasets.builder()
    config.dataset = "cifar10"

    config.stem = "CIFAR"
    config.model = "ResNet50"
    config.projector = "SimCLR"

    config.ntxent_temp = 0.5
    
    config.learning_rate = 1.0
    config.warmup_epochs = 10
    config.momentum = 0.9
    config.batch_size = 1024

    # set either num_epochs or num_steps to a positive number
    config.start_step = 0
    config.num_epochs = 250
    config.num_steps = -1

    config.cache_dataset = True
    config.half_precision = False
    
    config.linear_eval_step_freq = 300

    config.num_train_steps = -1
    config.steps_per_eval = -1
    config.log_every_steps = -1

    # set to -1 to disable
    config.save_projector_step_freq = 200

    config.restore_projector = False
    config.freeze_projector = True

    return config
