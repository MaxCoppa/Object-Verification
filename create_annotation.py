#!/usr/bin/env python3
"""
Executable script to prepare annotations for Object Verification.
"""

import time
from torch.utils.data import DataLoader

# Project modules
from dataset_annotation_preparation import prepare_annotation
from dataset_preloader import VeriImageDataset
from utils import image_utils

# Import project configuration
from config import ANNOTATIONS_PATH, RAW_ANNOTATIONS_PATH, DATA_PATH


def main(
    load=True, transform_type="test", train_ratio=0.5, n_error=2, n_augmentation=5
):
    """
    Prepare annotation from the dataset.

    Args:
        load (bool): Whether to load datasets images
        transform_type (str): Type of image transform for visualization
        train_ratio (float): Train/test split ratio
        n_error (int): Number of errors allowed during preprocessing
        n_augmentation (int): Number of augmentations per sample
    """

    # Paths
    raw_annotation_sharks = RAW_ANNOTATIONS_PATH / "raw_annotations_sharks.csv"
    preprocessed_annotation_sharks = (
        ANNOTATIONS_PATH / "preprocessed_annotations_sharks.csv"
    )
    images_dir = DATA_PATH / "animals"

    # -------------------------
    # Prepare data annotations
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

    # -------------------------
    # Load sample images
    # -------------------------
    if load:
        transform_default = image_utils.transform_fc(transform_type)

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

        # Combine train and test for inspection
        data = training_data + test_data

        dataloader = DataLoader(
            data, batch_size=2, shuffle=True, num_workers=0, pin_memory=True
        )

        for k, batch in enumerate(dataloader, start=1):
            print(f"batch: {k}")
            pass

        elapsed_time = time.time() - start_time
        print(f"Loading all the Data in : {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    main()
