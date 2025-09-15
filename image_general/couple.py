import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils import image_utils

from .image import Image
from .loader import ImagePreprocessor


class CoupleImage:
    """
    Represents a pair of related images (e.g., two views of the same object),
    and enables comparison or matching using a model.

    Attributes:
        img1 (Image): First image in the pair.
        img2 (Image): Second image in the pair.
        preprocessor (ImagePreprocessor): Reference for transforms, device, and model.
        prediction (int or None): Model output prediction label.
        score (float or None): Confidence score of the prediction.
    """

    def __init__(self, img1: Image, img2: Image, preprocessor: ImagePreprocessor):
        self.img1 = img1
        self.img2 = img2
        self.preprocessor = preprocessor

        self.prediction = None
        self.score = None

    def prepare(self):
        """
        Loads and transforms both images for model input.
        """
        self.img1.load_img()
        self.img2.load_img()

        self.img1.transform_img()
        self.img2.transform_img()

    def predict(self):
        """
        Performs a model prediction using the transformed image pair.
        """
        self.prepare()
        input1 = self.img1.img_transform.unsqueeze(0).to(self.preprocessor.device)
        input2 = self.img2.img_transform.unsqueeze(0).to(self.preprocessor.device)

        if not self.preprocessor.model:
            raise ValueError(
                "Model is not loaded. Please load or initialize the model before prediction."
            )

        with torch.no_grad():
            pred, score = self.preprocessor.model.predict(input1, input2)

        self.prediction = int(pred.item())
        self.score = float(score)
        return self.prediction, self.score

    def show(self):
        vis1 = self.img1.transform_img_visualise()
        vis2 = self.img2.transform_img_visualise()
        image_utils.image_plot.display_images(
            vis1,
            vis2,
            self.label,
            self.prediction,
            self.score,
        )

    @property
    def label(self):
        return int(self.img1.id == self.img2.id)
