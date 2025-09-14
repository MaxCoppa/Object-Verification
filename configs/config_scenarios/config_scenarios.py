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


dataset_default = {
    "pairing_type": "couples",
    "annotation_filename": "dataset_preprocessed_annotations.csv",
    "train_ratio": 0.8,
}


dataset_config = {
    "general": build_config(
        "general", file_type="png", crop_type="object", **dataset_default
    ),
}


eval_default = {
    # "pairing_type": "test",
    "annotation_filename": "eval_preprocessed_annotations.csv",
    "train_ratio": 0,
    "n_error": 1,
    "n_augmentation": 1,
}


eval_config = {
    "": build_config("", **eval_default),
}


test_default = {
    "pairing_type": "test",
    "annotation_filename": "test_preprocessed_annotations.csv",
    "train_ratio": 0,
    "n_error": 1,
    "train": False,
}


test_config = {
    "general": build_config("", file_type="png", crop_type="object", **test_default),
}


test_data_loader_default = {
    "annotation_filename": "dataloader_test_preprocessed_annotations.csv",
}


test_data_loader_config = {
    "": build_config("", **test_data_loader_default),
}
