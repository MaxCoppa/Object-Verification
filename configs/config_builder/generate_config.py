import os
from .config_data import data_config


def build_annotation_path(config_type: str, filename: str) -> str:
    return os.path.join(data_config[config_type]["annotation_dir"], filename)


def build_config(
    config_type: str,
    train: bool = None,
    file_type: str = None,
    crop_type: str = None,
    n_augmentation: int = None,
    n_error: int = None,
    pairing_type: str = None,
    train_ratio: float = None,
    annotation_filename: str = "test.csv",
) -> dict:

    if config_type not in data_config:
        raise ValueError(
            f"Unsupported dataset: '{config_type}'. "
            f"Choose from: {list(data_config.keys())}"
        )

    annotation_path = build_annotation_path(config_type, annotation_filename)

    preparation_block = {
        "preprocessed_annotation_path": annotation_path,
        **data_config[config_type]["preparation"],
    }

    data_block = {
        "annotations_file": annotation_path,
    }

    if pairing_type is not None:
        preparation_block["pairing_type"] = pairing_type
        data_block["pairing_type"] = pairing_type

    if file_type is not None:
        preparation_block["file_type"] = file_type
        data_block["file_type"] = file_type

    if crop_type is not None:
        preparation_block["crop_type"] = crop_type
        data_block["crop_type"] = crop_type

    if n_augmentation is not None:
        preparation_block["n_augmentation"] = n_augmentation

    if n_error is not None:
        preparation_block["n_error"] = n_error

    if train_ratio is not None:
        preparation_block["train_ratio"] = train_ratio

    if pairing_type is not None:
        preparation_block["pairing_type"] = pairing_type

    if train is not None:
        data_block["train"] = train

    return {
        "preparation": preparation_block,
        "data": data_block,
    }
