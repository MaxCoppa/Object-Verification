from ..config_builder import build_config

test_data_loader_default = {
    "annotation_filename": "dataloader_test_preprocessed_annotations.csv",
}


test_data_loader_config = {
    "": build_config("", **test_data_loader_default),
}
