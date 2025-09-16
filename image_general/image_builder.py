import torch
import pandas as pd
from utils import image_utils

from .image import Image
from .couple import CoupleImage
from .loader import ImagePreprocessor


class ImageFactory:
    """
    Factory class for creating Image and CoupleImage instances in a consistent way.

    Attributes:
        preprocessor (ImagePreprocessor): Shared preprocessor for applying transformations,
                                          model access, and configuration.
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def create_img(self, row_img, col_type=""):
        return Image(row_img=row_img, preprocessor=self.preprocessor, col_type=col_type)

    def create_couple(self, img1, img2):
        return CoupleImage(img1=img1, img2=img2, preprocessor=self.preprocessor)
