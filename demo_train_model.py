# %% -------------------------
# Import necessary libraries
# -------------------------
import time
import torch
import numpy as np
import os

from torch.utils.data import DataLoader
import torch.optim as optim

# Project modules
from dataset_annotation_preparation import prepare_annotation
from dataset_preloader import VeriImageDataset
from utils import image_utils, model_utils
from veri_models import ObjectVeriSiamese
from training_evaluation import train_veri_model

# Import project configuration
from config import (
    ANNOTATIONS_PATH,
    RAW_ANNOTATIONS_PATH,
    DATA_PATH,
    MODELS_PATH,
    LOGS_PATH,
)

# %% -------------------------
# Define parameters
# -------------------------
train_ratio = 0.5
n_error = 2
n_augmentation = 100

load = True
transform_type = "test"

raw_annotation_sharks = RAW_ANNOTATIONS_PATH / "raw_annotations_sharks.csv"
preprocessed_annotation_sharks = ANNOTATIONS_PATH / "train_annotations_sharks.csv"
images_dir = DATA_PATH / "animals"

# %% -------------------------
# Test Data Annotation
# -------------------------
df = prepare_annotation(
    raw_annotation_path=raw_annotation_sharks,
    images_dir=images_dir,
    preprocessed_annotation_path=preprocessed_annotation_sharks,
    train_ratio=train_ratio,
    n_augmentation=n_augmentation,
    n_error=n_error,
)

# %% -------------------------
# Training setup
# -------------------------
frozen = True
loss_name = "Contrastiveloss"
model_name = "2"
transform_type_train = "transform_data_aug"
transform_type_val = "test"

n_augmentation = 1
backbone = "mobilenet_v3_small"
batch_size = 64
num_epochs = 1

# Loss function
criterion = model_utils.get_loss_function(loss_name)

# Data transforms
transform_train = image_utils.transform_fc(transform_type_train)
transform_val = image_utils.transform_fc(transform_type_val)

# Datasets
train_data = VeriImageDataset(
    annotations_file=preprocessed_annotation_sharks,
    train=True,
    transform=transform_train,
    crop_type=None,
)

val_data = VeriImageDataset(
    annotations_file=preprocessed_annotation_sharks,
    train=False,
    transform=transform_val,
    crop_type=None,
)

print(len(train_data), len(val_data))

# %% -------------------------
# Dataloaders
# -------------------------
dataloaders = {
    "train": DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    ),
    "val": DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    ),
}

# %% -------------------------
# Model setup
# -------------------------
model = ObjectVeriSiamese(backbone=backbone, freeze_backbone=frozen)
if frozen:
    params = model.fc.parameters()
else:
    params = model.parameters()

optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)

# %% -------------------------
# Train model
# -------------------------
model = train_veri_model(
    model,
    criterion=criterion,
    optimizer=optimizer,
    dataloaders=dataloaders,
    num_epochs=num_epochs,
    freeze_backbone=frozen,
    save_path=MODELS_PATH / f"model_{model_name}.pth",
    log_filename=LOGS_PATH / f"log_{model_name}.log",
    log_to_console=True,
    verbose=True,
)

# %%
