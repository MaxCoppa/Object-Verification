import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
import itertools
from ..model_utils import make_prediction

from .image_loader import img_loader
from .image_transforms import transform_image

# Main Function


def visualise_images_df(
    df,
    n_images=20,
    file_type="png",
    crop_type="object",
    debug=True,
    transform_type="visualise",
):
    """
    Visualizes images from a DataFrame.

    """
    for _, row in itertools.islice(df.iterrows(), n_images):  # Efficient slicing
        img1, img2 = preprocessed_images(row, file_type, crop_type, transform_type)
        if img1 is None or img2 is None:
            if debug:
                print(f"Skipping row {row.name} due to preprocessing error.")
            continue

        display_images(
            img1,
            img2,
            label=int(row["label"]),
            pred=int(row["prediction"]),
            score=float(row["distance"]),
        )


# Utility Functions


def preprocessed_images(
    row,
    file_type="png",
    crop_type="object",
    transform_type="visualise",
):
    """
    Preprocesses image pairs from a DataFrame row.
    """
    try:
        img1 = img_loader(row, "img", file_type, crop_type)
        img1 = transform_image(img1, transform_type)
        img2 = img_loader(row, "couple", file_type, crop_type)
        img2 = transform_image(img2, transform_type)
        # Ensure images are in correct format for visualization
        img1 = process_image_plt(img1)
        img2 = process_image_plt(img2)

        return img1, img2
    except Exception as e:
        print(f"Error processing image for row {row.name}: {e}")
        return None, None  # Return None values to prevent crashes


def process_image_plt(img_tensor):
    """
    Converts a PyTorch tensor to a NumPy array for plotting.
    """
    if img_tensor.shape[0] == 3:  # Convert (C, H, W) → (H, W, C) if RGB
        img_tensor = img_tensor.permute(1, 2, 0)
    return img_tensor.squeeze()


def display_images(img1, img2, label, pred, score):
    """
    Displays a pair of images with label and prediction info.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img1, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Image 1")

    axes[1].imshow(img2, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Image 2")

    # Determine correctness of prediction
    is_correct = pred == label
    title = f"Label: {label}, Pred: {pred}, Score : {score:.2f}" + (
        " (Error)" if not is_correct else ""
    )
    plt.tight_layout()
    plt.suptitle(title, fontsize=10)

    plt.show()
