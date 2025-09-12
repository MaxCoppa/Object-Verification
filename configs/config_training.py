import torch
import numpy as np

dict_n_erros = {}
change_config = {}


train_strat_config = {
    f"comparaison_{i}": {
        "train": change_config[i].get("train", ""),
        "val": change_config[i].get("val", ""),
        "loss": change_config[i].get("loss", "Contrastiveloss"),
        "save_path": f"/comparaison/comparaison_{i}.pth",
        "log_filename": f"/comparaison/training_{i}.log",
        "transform_train": change_config[i].get(
            "transform_train", "transform_data_aug"
        ),
        "transform_val": change_config[i].get("transform_test", "test"),
        "n_augmentation": change_config[i].get("n_augmentation", 1),
        "frozen": change_config[i].get("frozen", False),
        "backbone": change_config[i].get("backbone", "resnet50"),
        "n_errors": change_config[i].get("n_errors", dict_n_erros),
        "batch_size": change_config[i].get("batch_size", 64),
    }
    for i in change_config.keys()
}
