import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt


from utils.image_utils import load_and_process_image, crop_image, process_image_plt


class VeriImageDatasetTest(Dataset):
    """
    PyTorch Dataset class for loading object image pairs during testing/evaluation.
    Each sample consists of a pair of images, their corresponding label, and paths.

    Args:
        annotations_file (str): Path to the CSV file with image paths and labels.
        transform (callable): Torchvision transform(s) to apply to the images.
        file_type (str): Type of image file (e.g., "jpg", "png").
        crop_type (str or None): Optional crop strategy to apply to the images.
    """

    def __init__(
        self,
        annotations_file,
        transform=None,
        file_type="jpg",
        crop_type=None,
    ):

        self.img_labels = pd.read_csv(annotations_file)

        self.transform = transform

        self.file_type = file_type
        self.crop_type = crop_type

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        row = self.img_labels.iloc[idx]
        label = row["label"]

        image = self.img_loader(row, "img")
        couple = self.img_loader(row, "couple")

        image_path = row["img_path"]
        couple_path = row["couple_path"]

        if self.transform:
            image = self.transform(image)
            couple = self.transform(couple)

        return image, couple, label, image_path, couple_path

    def img_loader(self, row, col):

        path = row[col + "_path"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found at: {path}")

        # if os.stat(path).st_size == 0:
        #     raise FileNotFoundError(f"Image not found at: {path}")

        dict_load = {
            "png": lambda path: load_and_process_image(
                path,
            ),
            "default": lambda path: read_image(path),
        }

        image_loader = dict_load.get(self.file_type, dict_load["default"])

        image = image_loader(path)

        if self.crop_type:
            image = crop_image(image, row[col + "_crop"])

        return image

    def visualize_images(self, n=5):
        """
        Visualizes a set of image pairs from the dataset.

        Args:
            n (int): The number of image pairs to visualize.
        """
        # Create a figure to display n pairs of images
        for i in range(n):
            img1, img2, _, _, _ = self[i]  # Get the i-th image pair
            img1 = process_image_plt(img1)
            img2 = process_image_plt(img2)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(img1, cmap="gray")
            axes[0].axis("off")
            axes[0].set_title("Image 1")

            axes[1].imshow(img2, cmap="gray")
            axes[1].axis("off")
            axes[1].set_title("Image 2")

            plt.tight_layout()
            plt.show()
