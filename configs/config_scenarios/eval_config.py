from ..config_builder import build_config

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
