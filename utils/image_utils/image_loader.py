import os
import numpy as np
import cv2
import torch
from torchvision.io import read_image


def load_image_file(path):
    """
    Loads an image file and adjusts its bit-depth if necessary.
    """
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    try:
        max_img = np.max(img)
        if max_img > 16383:
            return np.right_shift(img, 4)
        if max_img > 4095:
            return np.right_shift(img, 2)
        return img
    except:
        print(os.path.dirname(path))
        return img


def load_and_process_image(path):
    """
    Loads an image file and processes it according to a specified algorithm and color processing method.
    """
    img = load_image_file(path)

    return img


def crop_image(image, crop_params):
    """
    Crops an image based on the given parameters.
    """
    if isinstance(crop_params, str):
        top, left, height, width = map(
            int, crop_params.strip("()").strip("[]").split(",")
        )
    elif isinstance(crop_params, tuple) and len(crop_params) == 4:
        top, left, height, width = crop_params
    else:
        raise ValueError(
            "crop_params must be a tuple (top, left, height, width) or a string '(top, left, height, width)'."
        )

    if torch.is_tensor(image):
        image = image[:, left : width + left, top : top + height]
    elif isinstance(image, np.ndarray):
        image = image[left : width + left, top : top + height]

    else:
        raise TypeError("The image must be a PyTorch tensor or a NumPy array.")

    return image


def anonymise_image(image, crop_params):
    """
    Anonymize a objects based on crop parameters.
    """
    if isinstance(crop_params, str):
        top, left, height, width = map(
            int, crop_params.strip("()").strip("[]").split(",")
        )
    elif isinstance(crop_params, tuple) and len(crop_params) == 4:
        top, left, height, width = crop_params
        top, left, height, width = int(top), int(left), int(height), int(width)
    else:
        raise ValueError(
            "crop_params must be a tuple (top, left, height, width) or a string '(top, left, height, width)'."
        )

    if torch.is_tensor(image):
        m = image.max()
        image[:, left : width + left, top : top + height] = m

    elif isinstance(image, np.ndarray):
        m = image.max()
        image[left : width + left, top : top + height] = m

    else:
        raise TypeError("The image must be a PyTorch tensor or a NumPy array.")

    return image


def img_loader(row, col, file_type=None, crop_type=None):
    """
    Loads and processes an image based on the file type and crop type.
    """
    img_path = row[col + "_path"]
    crop = None

    if crop_type:
        crop = row[col + "_crop"]

    image = load_image_file_generic(
        img_path,
        file_type=file_type,
        crop=crop,
    )

    return image


def load_image_file_generic(
    img_path,
    file_type=None,
    crop=None,
):
    """
    Loads and processes an image.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at: {img_path}")

    if file_type == "png":
        image = load_and_process_image(img_path)
    else:
        image = read_image(img_path)

    if crop:
        image = crop_image(image, crop)

    return image


def pad_to_square(image):
    """
    Pads an image (tensor or ndarray) to make it square by adding black borders.
    Returns the padded image.
    """
    if torch.is_tensor(image):
        _, h, w = image.shape
        c = max(h, w)
        padded = torch.zeros(
            (image.shape[0], c, c), dtype=image.dtype, device=image.device
        )
        y_offset = (c - h) // 2
        x_offset = (c - w) // 2
        padded[:, y_offset : y_offset + h, x_offset : x_offset + w] = image
        return padded

    elif isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        c = max(h, w)
        if image.ndim == 3:
            padded = np.zeros((c, c, image.shape[2]), dtype=image.dtype)
        else:
            padded = np.zeros((c, c), dtype=image.dtype)
        y_offset = (c - h) // 2
        x_offset = (c - w) // 2
        padded[y_offset : y_offset + h, x_offset : x_offset + w] = image
        return padded

    else:
        raise TypeError("The image must be a PyTorch tensor or a NumPy array.")
