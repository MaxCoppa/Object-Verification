# %%
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
    prepare_dataset,
)

from torch.utils.data import DataLoader
import torch.optim as optim

from configs import train_strat_config, build_train_config, get_config

training_config = get_config("training_config")
val_config = get_config("test_config")
test_config = get_config("test_config")

dataset_config = get_config("dataset_config")


# """
# Training Basic
# """


# %%
train_config_list = ""
val_config_list = ""

frozen = False

dict_n_errors = {
    "general": 2,
}

loss_name = "Contrastiveloss"

transform_train = "transform_data_aug"
transform_val = "test"

n_augmentation = 1
backbone = "resnet50"
batch_size = 64
num_epochs = 1
# %%

criterion = model_utils.get_loss_function(loss_name)
train_data_list = []
# %%
for config_type_train in train_config_list:
    prepare_dataset(
        dataset_name=preparation_type,
        n_error=dict_n_errors[config_type_train],
        **training_config[config_type_train]["preparation"]
    )
    transform = image_utils.transform_fc(transform_train)
    train_data = VeriImageDataset(
        transform=transform, **training_config[config_type_train]["data"]
    )
    # train_data.visualize_images(n=1)
    print(config_type_train, len(train_data))
    train_data_list.append(train_data)

train_data = torch.utils.data.ConcatDataset(train_data_list)

test_data_list = []

for config_type_val in val_config_list:

    prepare_dataset(
        dataset_name=preparation_type, **val_config[config_type_val]["preparation"]
    )

    transform = image_utils.transform_fc(transform_val)

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
        batch_size=batch_siz,
        shuffle=True,
        num_workers=24,
        pin_memory=True,
    ),
    "val": DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=24,
        pin_memory=True,
    ),
}

model = ObjectVeriSiamese(backbone=backbone, freeze_backbone=frozen)
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
    num_epochs=num_epochs,
    freeze_backbone=frozen,
    save_path=config["save_path"],
    log_filename=config["log_filename"],
    log_to_console=False,
    verbose=False,
)

# %%
