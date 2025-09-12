# %% Import necessary libraries
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset_annotation_preparation import prepare_dataset
from dataset_preloader import (
    VeriImageDataset,
)

from utils import config_utils, image_utils
from configs import build_config

# %% Define parameters
preparation_type = "general"
file_type = "jpg"
crop_type = None
pairing_type = "couples"

train_ratio = 0.8
n_error = 1
n_augmentation = 1

load = False
transform_type = "x"

annotation_filename = "test.csv"

# %% Build the configuration for the data annotation preparation

data_configs = build_config(
    config_type=preparation_type,
    train=None,
    file_type=file_type,
    crop_type=crop_type,
    n_augmentation=n_augmentation,
    n_error=n_error,
    pairing_type=pairing_type,
    train_ratio=train_ratio,
    annotation_filename=annotation_filename,
)

config_preparation_annot = data_configs["preparation"]
config_data_loader = data_configs["data"]


# %% Test Data Annotation
start = time.time()
df = prepare_dataset(
    dataset_name=preparation_type,
    **config_preparation_annot,
)

print(f"Annotation preparation time: {time.time() - start:.4f} seconds")
print(f"Annotation DataFrame shape: {df.shape}, Columns : {list(df.columns)}")
print(
    f"Number of data : {df.shape[0]}, Data Training : {df['train'].sum()}, Test Data : {(1 - df['train']).sum()}"
)
print(
    f"Positive Pairs : {df['label'].sum()}, Negative Pairs : {(1 - df['label']).sum()}"
)


# %% Test data Loading

# # %% Visualize sample images from the training dataset

# start = time.time()
# test_data.visualize_images(n=10)
# print(f"Image visualization time: {time.time() - start:.4f} seconds")

if load:

    transform_default = image_utils.transform_fc(transform_type)

    training_data = VeriImageDataset(
        train=True,
        transform=transform_default,
        **config_data_loader,
    )

    test_data = VeriImageDataset(
        train=False,
        transform=transform_default,
        **config_data_loader,
    )

    start_time = time.time()

    data = training_data + test_data

    dataloader = DataLoader(
        data, batch_size=128, shuffle=True, num_workers=8, pin_memory=True
    )

    for batch in tqdm(dataloader, desc="Batches preprocessed"):

        pass

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Loading all the Data in : {elapsed_time:.4f} secondes")

# %%
