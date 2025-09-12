from .config_scenarios import (
    dataset_config,
    eval_config,
    test_config,
    training_config,
    test_data_loader_config,
)


def get_config(config_type: str):

    dict_configs = {
        "dataset_config": dataset_config,
        "eval_config": eval_config,
        "test_config": test_config,
        "training_config": training_config,
        "test_data_loader_config": test_data_loader_config,
    }

    if config_type not in dict_configs:
        raise ValueError(
            f"Invalid config_type: '{config_type}'. Valid options are: {list(dict_configs.keys())}"
        )

    return dict_configs[config_type]
