"""
Executable script to train the Object Verification Siamese Network.
Supports command-line arguments with short flags.
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
        "-f",
        "--frozen",
        type=bool,
        default=True,
        help="Freeze backbone during training",
    )
    parser.add_argument(
        "-r",
        "--train_ratio",
        type=float,
        default=0.5,
        help="Train/validation split ratio",
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
    return parser.parse_args()


def main(
    epochs=1,
    backbone="resnet50",
    frozen=True,
    train_ratio=0.5,
    augmentation=100,
    loss="Contrastiveloss",
):
    """
    Train the Siamese Object Verification model.

    Args:
        epochs (int): Number of training epochs
        backbone (str): Backbone model
        frozen (bool): Freeze backbone during training
        train_ratio (float): Train/validation split ratio
        augmentation (int): Number of augmentations per sample
        loss (str): Loss function name
    """
    # -------------------------
    # Dataset paths
    # -------------------------
    raw_annotation_sharks = RAW_ANNOTATIONS_PATH / "raw_annotations_sharks.csv"
    preprocessed_annotation_sharks = ANNOTATIONS_PATH / "train_annotations_sharks.csv"
    images_dir = DATA_PATH / "animals"

    # -------------------------
    # Prepare data annotations
    # -------------------------
    df = prepare_annotation(
        raw_annotation_path=raw_annotation_sharks,
        images_dir=images_dir,
        preprocessed_annotation_path=preprocessed_annotation_sharks,
        train_ratio=train_ratio,
        n_augmentation=augmentation,
        n_error=2,
    )

    # -------------------------
    # Setup training
    # -------------------------
    criterion = model_utils.get_loss_function(loss)
    transform_train = image_utils.transform_fc("transform_data_aug")
    transform_val = image_utils.transform_fc("test")

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

    dataloaders = {
        "train": DataLoader(
            train_data,
            batch_size=64,  # Fixed batch size
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_data,
            batch_size=64,  # Fixed batch size
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
        save_path=MODELS_PATH / "model_1.pth",
        log_filename=LOGS_PATH / "log_1.log",
        log_to_console=True,
        verbose=True,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        epochs=args.epochs,
        backbone=args.backbone,
        frozen=args.frozen,
        train_ratio=args.train_ratio,
        augmentation=args.augmentation,
        loss=args.loss,
    )
