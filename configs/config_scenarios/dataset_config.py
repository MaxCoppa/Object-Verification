from ..config_builder import build_config

dataset_default = {
    "pairing_type": "couples",
    "annotation_filename": "dataset_preprocessed_annotations.csv",
    "train_ratio": 0.8,
}


dataset_config = {
    "general": build_config(
        "general", file_type="png", crop_type="car", **dataset_default
    ),
}
