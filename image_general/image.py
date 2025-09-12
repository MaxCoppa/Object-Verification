import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils import image_utils

from .loader import ImagePreprocessor


class Image:
    """
    Represents a single annotated image with support for preprocessing, transformation,
    visualization, and metadata access.

    Attributes:
        preprocessor (ImagePreprocessor): Reference to the preprocessor containing config and transforms.
        row_img (pd.Series): Annotation row describing the image (paths, crops, metadata).
        img (np.ndarray or None): Loaded image.
        col_type (str): Column prefix to distinguish between different image types (e.g., 'front', 'rear').
    """

    def __init__(
        self, preprocessor: ImagePreprocessor, row_img: pd.Series, col_type=""
    ):
        self.preprocessor = preprocessor
        self.row_img = row_img.copy()
        self.img = None
        self.col_type = col_type
        self.process_row()

    def process_row(self):
        path = image_utils.create_image_path(
            row_img=self.row_img,
            file_type=self.preprocessor.file_type,
            col_type=self.col_type,
        )
        if not image_utils.validate_correct_img_path(path):
            raise FileNotFoundError(f"Image not found at path: {path}")
        self.row_img[self.col_type + "path"] = path

    def load_img(self):
        self.img = image_utils.load_image_file_generic(
            self.row_img[self.col_type + "path"],
            algo_pair=self.preprocessor.algo_pair,
            file_type=self.preprocessor.file_type,
            crop=self.row_img.get(self.col_type + "crop"),
        )
        return self.img

    def transform_img(self):
        if self.img is None:
            raise RuntimeError("Image must be loaded before being transformed.")
        if self.preprocessor.transform is None:
            raise ValueError("No transform function provided in preprocessor.")
        self.img_transform = self.preprocessor.transform(self.img)
        return self.img_transform

    def transform_img_visualise(self):

        self.img_visualise = image_utils.transform_fc("same")(self.img)

        if self.preprocessor.anonymize_plate & (
            (self.preprocessor.file_type == "png")
            | (self.preprocessor.file_type == "plate")
        ):
            if self.preprocessor.crop_type:
                crop_params = ""
            else:
                crop_params = ""
            self.img_visualise = image_utils.anonymise_image(
                self.img_visualise, crop_params
            )

        self.img_visualise = image_utils.process_image_plt(self.img_visualise)
        return self.img_visualise

    def show_img(self):
        self.transform_img_visualise()

        plt.imshow(self.img_visualise, cmap="gray")
        plt.axis("off")
        plt.title(self.plate)
        plt.show()

    # -------- Metadata Accessors -------- #
    @property
    def plate(self):
        return self.row_img.get(self.col_type + "plate")

    @property
    def path(self):
        return self.row_img.get(self.col_type + "path")

    @property
    def uid(self):
        return self.row_img.get(self.col_type + "UID")

    @property
    def tsp(self):
        return pd.to_datetime(self.row_img.get(self.col_type + "tsp"), format="mixed")

    @property
    def folder(self):
        return self.row_img.get(self.col_type + "folder")

    @property
    def hour(self):
        return pd.to_datetime(self.row_img[self.col_type + "tsp"], format="mixed").hour
