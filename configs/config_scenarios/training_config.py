from ..config_builder import build_config

train_default = {
    "pairing_type": "couples",
    "annotation_filename": "train_preprocessed_annotations.csv",
    "train_ratio": 1,
    "train": True,
}


training_config = {
    "": build_config(
        "general",
        file_type="png",
        crop_type="object",
        n_augmentation=2,
        **train_default
    ),
}
