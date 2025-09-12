# %% Import necessary libraries
import time

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
from torch.utils.data import random_split
import matplotlib.pyplot as plt

from dataset_annotation_preparation import (
    prepare_annotation,
)
from dataset_preloader import (
    VeriImageDataset,
)

from utils import config_utils, image_utils
from configs import get_config

# %% Load Config Files
data_preparation_fc = {
    "general": prepare_annotation,
}

test_data_loader_configs = get_config("test_data_loader_config")

# %% Define parameters
preparation_type = "general"
file_type = "jpg"
crop_type = "car"
pairing_type = "test"

train_ratio = 0
n_error = 1
n_augmentation = 1


# %% Retrieve configuration values and select preparation function
config_preparation_annot = test_data_loader_configs[preparation_type]["preparation"]
config_data_loader = test_data_loader_configs[preparation_type]["data"]
preparation_method = data_preparation_fc[preparation_type]

# %% Test Data Annotation
start = time.time()
df = preparation_method(
    file_type=file_type,
    crop_type=crop_type,
    pairing_type=pairing_type,
    train_ratio=train_ratio,
    n_error=n_error,
    n_augmentation=n_augmentation,
    **config_preparation_annot,
)

print(f"Annotation preparation time: {time.time() - start:.4f} seconds")
print(f"Annotation DataFrame shape: {df.shape}, Columns : {list(df.columns)}")

# %% Test data Loading
transform_default = image_utils.transform_fc("visualise")

training_data = VeriImageDataset(
    train=True,
    transform=transform_default,
    file_type=file_type,
    crop_type=crop_type,
    pairing_type=pairing_type,
    **config_data_loader,
)

test_data = VeriImageDataset(
    train=False,
    transform=transform_default,
    file_type=file_type,
    crop_type=crop_type,
    pairing_type=pairing_type,
    **config_data_loader,
)

n = len(training_data) + len(test_data)
print(
    f"Number of data : {n}, training_data : {len(training_data)}, test_data {len(test_data)}"
)

# %% Visualize sample images from the training dataset
start = time.time()
test_data.visualize_images(n=10)
print(f"Image visualization time: {time.time() - start:.4f} seconds")

# %% Test Batch Data Loader
start_time = time.time()

data = training_data + test_data

dataloader = DataLoader(
    data, batch_size=128, shuffle=True, num_workers=8, pin_memory=True
)
k = 0

for batch in dataloader:

    k += 1
    print("batch: " + str(k))
    pass


end_time = time.time()

elapsed_time = end_time - start_time
print(
    f"Temps de chargement du DataFrame dans le DataLoader : {elapsed_time:.4f} secondes"
)

# %%
