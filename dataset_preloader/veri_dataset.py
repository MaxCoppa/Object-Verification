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


class VeriImageDataset(Dataset):
    """
    Custom PyTorch Dataset for object Verification using image pairs or triplets.
    Supports training and evaluation using annotations that define correct (positive) and incorrect (negative) object matches.

    Args:
        annotations_file (str): Path to the CSV file containing image paths and metadata.
        correct_pair_ratio (float): Probability of sampling a positive pair in triplet mode. Default is 0.5.
        img_dir (str): Optional base directory for images.
        train (bool): Whether to load training data (True) or testing data (False).
        transform (callable): Optional torchvision transform to apply to images.
        file_type (str): Type of image file ("jpg", "png", etc.).
        pairing_type (str): Pairing mode - "pairs" for direct pairs, "triplets" for triplet generation.
        crop_type (str or None): Crop method to apply if cropping is enabled.
    """

    def __init__(
        self,
        annotations_file,
        correct_pair_ratio: int = 0.5,
        img_dir=None,
        train=True,
        transform=None,
        file_type="jpg",
        pairing_type="couples",
        crop_type=None,
    ):

        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.train = train

        self.img_labels = self.img_labels[self.img_labels["train"] == int(self.train)]

        self.transform = transform

        self.correct_pair_ratio = correct_pair_ratio
        self.file_type = file_type
        self.pairing_type = pairing_type
        self.crop_type = crop_type

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        row = self.img_labels.iloc[idx]
        image = self.img_loader(row, "img")

        if self.pairing_type == "triplets":
            if np.random.random() <= self.correct_pair_ratio:
                label = 1
                couple = self.img_loader(row, "couple")
            else:
                label = 0
                couple = self.img_loader(row, "error")

        else:
            label = row["label"]
            couple = self.img_loader(row, "couple")

        if self.transform:
            image = self.transform(image)
            couple = self.transform(couple)
        return image, couple, label

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
            img1, img2, _ = self[i]  # Get the i-th image pair
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
