# %% -------------------------
# Import necessary libraries
# -------------------------
import time
from torch.utils.data import DataLoader

# Project modules
from dataset_annotation_preparation import prepare_annotation
from dataset_preloader import VeriImageDataset
from utils import image_utils

# Import project configuration
from config import (
    ANNOTATIONS_PATH,
    RAW_ANNOTATIONS_PATH,
    DATA_PATH,
)

# %% -------------------------
# Define parameters
# -------------------------
crop_type = None
train_ratio = 0.5
n_error = 2
n_augmentation = 5

load = True
transform_type = "test"

raw_annotation_sharks = RAW_ANNOTATIONS_PATH / "raw_annotations_sharks.csv"
preprocessed_annotation_sharks = (
    ANNOTATIONS_PATH / "preprocessed_annotations_sharks.csv"
)
images_dir = DATA_PATH / "animals"

# %% -------------------------
# Test Data Annotation
# -------------------------
start = time.time()

df = prepare_annotation(
    raw_annotation_path=raw_annotation_sharks,
    images_dir=images_dir,
    preprocessed_annotation_path=preprocessed_annotation_sharks,
    train_ratio=train_ratio,
    n_augmentation=n_augmentation,
    n_error=n_error,
)

print(f"Annotation preparation time: {time.time() - start:.4f} seconds")
print(f"Annotation DataFrame shape: {df.shape}, Columns : {list(df.columns)}")
print(
    f"Number of data : {df.shape[0]}, Data Training : {df['train'].sum()}, Test Data : {(1 - df['train']).sum()}"
)
print(
    f"Positive Pairs : {df['label'].sum()}, Negative Pairs : {(1 - df['label']).sum()}"
)

# %% -------------------------
# Visualize sample images from the training dataset
# -------------------------
transform_default = image_utils.transform_fc(transform_type)

if load:
    # Build datasets
    training_data = VeriImageDataset(
        annotations_file=preprocessed_annotation_sharks,
        train=True,
        transform=transform_default,
        crop_type=None,
    )

    test_data = VeriImageDataset(
        annotations_file=preprocessed_annotation_sharks,
        train=False,
        transform=transform_default,
        crop_type=None,
    )

    start_time = time.time()

    # Combine train and test for quick inspection
    data = training_data + test_data

    # Small batch size to check data loading pipeline
    dataloader = DataLoader(
        data, batch_size=2, shuffle=True, num_workers=0, pin_memory=True
    )  # Note: num_workers=0 for MacOS compatibility

    k = 0
    for batch in dataloader:
        k += 1
        print("batch: " + str(k))
        pass

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Loading all the Data in : {elapsed_time:.4f} seconds")

# %% -------------------------
# Visualize images
# -------------------------
start = time.time()
test_data.visualize_images(n=2)
print(f"Image visualization time: {time.time() - start:.4f} seconds")

# %%
