import torch
import numpy as np
import os

from utils import image_utils, model_utils

from dataset_preloader import VeriImageDataset

from veri_models import ObjectVeriSiamese

from training_evaluation import (
    train_veri_model,
)


from dataset_annotation_preparation import (
    prepare_annotation,
)

from torch.utils.data import DataLoader
import torch.optim as optim

from configs import train_strat_config, get_config

data_preparation_fc = {
    "general": prepare_annotation,
}


training_config = get_config("training_config")
val_config = get_config("test_config")
test_config = get_config("test_config")

dataset_config = get_config("dataset_config")


# """
# Training Basic
# """

loss_eq = {
    "Contrastiveloss": model_utils.losses.ContrastiveLoss(margin=1),
    "CircleLoss": model_utils.losses.CircleLoss(m=0.25, gamma=1),
    "FocalLoss": model_utils.losses.FocalLoss(alpha=0.25, gamma=2.0),
    "FocalLoss_v2": model_utils.losses.FocalLoss(alpha=0.25, gamma=1.0),
    "FocalLoss_v3": model_utils.losses.FocalLoss(alpha=0.4, gamma=2.0),
    "FocalLoss_v5": model_utils.losses.FocalLoss(alpha=0.4, gamma=10.0),
    "FocalLoss_v4": model_utils.losses.FocalLoss(alpha=0.4, gamma=1.0),
    "CosineLoss": model_utils.losses.CosineLoss(margin=0.2),
    "ContrastiveLossBis": model_utils.losses.ContrastiveLoss(margin=np.sqrt(2)),
    "ContrastiveLossBatches": model_utils.losses.ContrastiveLossBatches(margin=1),
}


for key, config in train_strat_config.items():

    train_config_list = config["train"].split(":")
    val_config_list = config["val"].split(":")
    frozen = config["frozen"]
    dict_n_errors = config["n_errors"]

    criterion = loss_eq[config["loss"]]
    train_data_list = []

    for config_type_train in train_config_list:
        data_preparation_fc[config_type_train](
            n_error=dict_n_errors[config_type_train],
            **training_config[config_type_train]["preparation"]
        )
        transform = image_utils.transform_fc(config["transform_train"])
        train_data = VeriImageDataset(
            transform=transform, **training_config[config_type_train]["data"]
        )
        # train_data.visualize_images(n=1)
        print(config_type_train, len(train_data))
        train_data_list.append(train_data)

    train_data = torch.utils.data.ConcatDataset(train_data_list)

    test_data_list = []

    for config_type_val in val_config_list:

        data_preparation_fc[config_type_val](
            **val_config[config_type_val]["preparation"]
        )

        transform = image_utils.transform_fc(config["transform_val"])

        val_data = VeriImageDataset(
            transform=transform, **val_config[config_type_val]["data"]
        )

        # val_data.visualize_images(n=1)
        print(config_type_val, len(val_data))
        test_data_list.append(val_data)

    val_data = torch.utils.data.ConcatDataset(test_data_list)
    print(len(train_data), len(val_data))
    dataloaders = {
        "train": DataLoader(
            train_data,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=24,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_data,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=24,
            pin_memory=True,
        ),
    }

    model = ObjectVeriSiamese(backbone=config["backbone"], freeze_backbone=frozen)
    if frozen:
        params = model.fc.parameters()

    else:
        params = model.parameters()

    optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)
    model = train_veri_model(
        model,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders=dataloaders,
        num_epochs=1,
        freeze_backbone=frozen,
        save_path=config["save_path"],
        log_filename=config["log_filename"],
        log_to_console=False,
        verbose=False,
    )

"""
Training New
"""

config_train = {
    "transform_train": "transform_data_aug",
}
config = build_train_config(config_train, "test_last_model")

train_config_list = config["train"].split(":")
val_config_list = config["val"].split(":")
frozen = config["frozen"]
dict_n_errors = config["n_errors"]

criterion = loss_eq[config["loss"]]
train_data_list = []

for config_type_train in train_config_list:
    data_preparation_fc[config_type_train](
        n_error=dict_n_errors[config_type_train],
        **training_config[config_type_train]["preparation"]
    )
    transform = image_utils.transform_fc(config["transform_train"])
    train_data = VeriImageDataset(
        transform=transform, **training_config[config_type_train]["data"]
    )
    train_data_list.append(train_data)

    # train_data.visualize_images(n=1)
    print(config_type_train, len(train_data))

test_data_list = []

for config_type_val in val_config_list:
    dataset_config[config_type_val]["preparation"]["train_ratio"] = 0.2
    data_preparation_fc[config_type_val](
        n_error=1, n_augmentation=1, **dataset_config[config_type_val]["preparation"]
    )

    transform = image_utils.transform_fc(config["transform_train"])
    train_data = VeriImageDataset(
        transform=transform, train=True, **dataset_config[config_type_val]["data"]
    )
    # train_data.visualize_images(n=1)
    print(config_type_val, len(train_data))
    train_data_list.append(train_data)

    transform = image_utils.transform_fc(config["transform_train"])
    val_data = VeriImageDataset(
        transform=transform, train=False, **dataset_config[config_type_val]["data"]
    )
    # val_data.visualize_images(n=1)
    print(config_type_val, len(val_data))
    test_data_list.append(val_data)

train_data = torch.utils.data.ConcatDataset(train_data_list)
val_data = torch.utils.data.ConcatDataset(test_data_list)

print(len(train_data), len(val_data))
dataloaders = {
    "train": DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=24,
        pin_memory=True,
    ),
    "val": DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=24,
        pin_memory=True,
    ),
}

model = ObjectVeriSiamese(backbone=config["backbone"], freeze_backbone=frozen)
if frozen:
    params = model.fc.parameters()

else:
    params = model.parameters()

optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)
model = train_veri_model(
    model,
    criterion=criterion,
    optimizer=optimizer,
    dataloaders=dataloaders,
    num_epochs=1,
    freeze_backbone=frozen,
    save_path=config["save_path"],
    log_filename=config["log_filename"],
    log_to_console=False,
    verbose=False,
)
