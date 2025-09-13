from ..config_builder import build_config

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
