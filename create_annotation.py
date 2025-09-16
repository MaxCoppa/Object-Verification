"""
Executable script to prepare dataset annotations for Object Verification tasks.

This script:
1. Preprocesses raw annotations into a structured CSV format.
2. Splits the dataset into training and test sets.
3. Applies augmentation to increase sample diversity.
4. Optionally loads and verifies dataset samples via DataLoader.

Typical usage:
    python prepare_annotations.py -l True -r 0.7 -e 2 -a 5
"""

import argparse
import time
from torch.utils.data import DataLoader

# Project modules
from dataset_annotation_preparation import prepare_annotation
from dataset_preloader import VeriImageDataset
from utils import image_utils

# Import project configuration
from config import ANNOTATIONS_PATH, RAW_ANNOTATIONS_PATH, DATA_PATH


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset annotations for Object Verification"
    )

    parser.add_argument(
        "-l",
        "--load",
        type=bool,
        default=False,
        help="Load images after annotation preparation",
    )
    parser.add_argument(
        "-r", "--train_ratio", type=float, default=0.5, help="Train/test split ratio"
    )
    parser.add_argument(
        "-e",
        "--n_error",
        type=int,
        default=2,
        help="Number of errors allowed during preprocessing",
    )
    parser.add_argument(
        "-a",
        "--n_augmentation",
        type=int,
        default=5,
        help="Number of augmentations per sample",
    )
    return parser.parse_args()


def main(load=True, train_ratio=0.5, n_error=2, n_augmentation=5):
    """
    Prepare annotation from the dataset.

    Args:
        load (bool): Whether to load dataset images
        transform_type (str): Type of image transform for visualization
        train_ratio (float): Train/test split ratio
        n_error (int): Number of errors allowed during preprocessing
        n_augmentation (int): Number of augmentations per sample
    """
    transform_type = "test"
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
    elapsed = time.time() - start
    print(f"\n=== Annotation Preparation Summary ===")
    print(f"Preparation time: {elapsed:.4f} seconds")
    print(f"Annotation DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Number of data: {df.shape[0]}")
    print(f"Training data: {df['train'].sum()}, Test data: {(1-df['train']).sum()}")
    print(
        f"Positive pairs: {df['label'].sum()}, Negative pairs: {(1-df['label']).sum()}\n"
    )

    # -------------------------
    # Load sample images (optional)
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
        data = training_data + test_data
        dataloader = DataLoader(
            data, batch_size=2, shuffle=True, num_workers=0, pin_memory=True
        )

        for k, batch in enumerate(dataloader, start=1):
            print(f"Batch {k}: loaded")
            pass

        elapsed_time = time.time() - start_time
        print(f"\nLoading all images took: {elapsed_time:.4f} seconds\n")


if __name__ == "__main__":
    args = parse_args()
    main(
        load=args.load,
        train_ratio=args.train_ratio,
        n_error=args.n_error,
        n_augmentation=args.n_augmentation,
    )
