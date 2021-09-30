import ml_collections


def get_clf_config() -> ml_collections.ConfigDict:
    """Get the default hyperparameter configuration."""
    clf_config = ml_collections.ConfigDict()

    clf_config.learning_rate = 1e-3
    clf_config.batch_size = 2048
    clf_config.cache_dataset = True
    clf_config.start_step = 0
    clf_config.num_epochs = 1
    clf_config.num_steps = -1

    # l2 regularizer coefficient
    clf_config.minl2coeff = 0.0
    clf_config.maxl2coeff = 1e-3
    clf_config.num_heads = 10

    return clf_config


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.clf_config = get_clf_config()

    config.name = "supervised"

    # `name` argument of tensorflow_datasets.builder()
    config.dataset = "cifar10"

    config.stem = "CIFAR"
    config.model = "ResNet50"
    config.projector = "SimCLR"

    # train with a supervised loss, make sure an appropriate projector
    # is selected if setting this to True.
    config.is_supervised = False

    config.ntxent_temp = 0.5
    config.ntxent_unif_coeff = 1.0

    config.learning_rate = 1.0
    config.warmup_epochs = 10
    config.momentum = 0.9
    config.batch_size = 2048

    # set either num_epochs or num_steps to a positive number
    config.start_step = 0
    config.num_epochs = 300
    config.num_steps = -1

    config.cache_dataset = True
    config.half_precision = False

    config.linear_eval_freq = 300

    config.num_train_steps = -1
    config.steps_per_eval = -1
    config.log_every_steps = -1

    # set to -1 to disable
    config.checkpoint_freq = 1000
    config.num_checkpoints_to_keep = 10

    config.restore_projector = ""
    config.freeze_projector = False

    return config
