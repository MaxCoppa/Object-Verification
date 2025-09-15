"""
Executable script to train the Siamese Network for Object Verification.

This script:
1. Prepares raw dataset annotations for training and validation.
2. Constructs PyTorch Datasets and DataLoaders for both splits.
3. Builds a Siamese model with a configurable backbone.
4. Trains the model with a specified loss function and optimizer.
5. Saves the trained model weights and logs training progress.

Typical usage:
    python train_siamese.py -e 10 -b resnet50 -a 100 -l Contrastiveloss -n shark_v1
"""

import argparse
import torch
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


def parse_args():
    """Parse command-line arguments with short flags."""
    parser = argparse.ArgumentParser(
        description="Train Siamese Object Verification model."
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "-b", "--backbone", type=str, default="resnet50", help="Backbone model"
    )

    parser.add_argument(
        "-a",
        "--augmentation",
        type=int,
        default=100,
        help="Number of augmentations per sample",
    )
    parser.add_argument(
        "-l", "--loss", type=str, default="Contrastiveloss", help="Loss function name"
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        default="shark_v1",
        help="Name for saved model and log files",
    )
    parser.add_argument(
        "-tt",
        "--transform_train",
        type=str,
        default="transform_data_aug",
        help="Type of transform for training data",
    )
    parser.add_argument(
        "-tv",
        "--transform_val",
        type=str,
        default="test",
        help="Type of transform for validation data",
    )
    return parser.parse_args()


def main(
    epochs=1,
    backbone="resnet50",
    augmentation=100,
    loss="Contrastiveloss",
    model_name="model_1",
    transform_train_type="transform_data_aug",
    transform_val_type="test",
):
    """
    Train the Siamese Object Verification model.

    Args:
        epochs (int): Number of training epochs
        backbone (str): Backbone model
        frozen (bool): Freeze backbone during training
        augmentation (int): Number of augmentations per sample
        loss (str): Loss function name
        model_name (str): Name for saved model and log files
        transform_train_type (str): Transform type for training
        transform_val_type (str): Transform type for validation
    """
    # -------------------------
    # Dataset paths
    # -------------------------
    raw_annotation_sharks = RAW_ANNOTATIONS_PATH / "raw_annotations_sharks.csv"
    preprocessed_annotation_sharks = ANNOTATIONS_PATH / "train_annotations_sharks.csv"
    images_dir = DATA_PATH / "animals"

    frozen = True
    # -------------------------
    # Prepare data annotations
    # -------------------------
    df = prepare_annotation(
        raw_annotation_path=raw_annotation_sharks,
        images_dir=images_dir,
        preprocessed_annotation_path=preprocessed_annotation_sharks,
        train_ratio=0.5,
        n_augmentation=augmentation,
        n_error=2,
    )

    # -------------------------
    # Setup training
    # -------------------------
    criterion = model_utils.get_loss_function(loss)
    transform_train = image_utils.transform_fc(transform_train_type)
    transform_val = image_utils.transform_fc(transform_val_type)

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

    print(f"\n=== Training Summary ===")
    print(f"Dataset: Shark dataset")
    print(f"Total annotations: {len(df)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Training epochs: {epochs}")
    print(f"Backbone: {backbone}")
    print(f"Loss function: {loss}")
    print(f"Model name: {model_name}\n")

    dataloaders = {
        "train": DataLoader(
            train_data,
            batch_size=64,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_data,
            batch_size=64,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        ),
    }

    model = ObjectVeriSiamese(backbone=backbone, freeze_backbone=frozen)
    params = model.fc.parameters() if frozen else model.parameters()
    optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)

    # -------------------------
    # Train the model
    # -------------------------
    model = train_veri_model(
        model,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders=dataloaders,
        num_epochs=epochs,
        freeze_backbone=frozen,
        save_path=MODELS_PATH / f"model_{model_name}.pth",
        log_filename=LOGS_PATH / f"log_{model_name}.log",
        log_to_console=True,
        verbose=True,
    )

    print(f"\n=== Training Completed ===")
    print(f"Model saved at: {MODELS_PATH / f'model_{model_name}.pth'}")
    print(f"Training log saved at: {LOGS_PATH / f'log_{model_name}.log'}\n")


if __name__ == "__main__":
    args = parse_args()
    main(
        epochs=args.epochs,
        backbone=args.backbone,
        augmentation=args.augmentation,
        loss=args.loss,
        model_name=args.model_name,
        transform_train_type=args.transform_train,
        transform_val_type=args.transform_val,
    )
