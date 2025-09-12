import torchvision.transforms.v2 as v2
import torch
import torchvision.transforms.functional as F


def transform_image(image, transform_type):
    """
    Applies a specific transformation to an image.
    """
    transform_dict = {
        "test": v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((254, 254)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        "visualise": v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((254, 254)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),

    }
    transform = transform_dict.get(transform_type, transform_dict["visualise"])

    return transform(image)


def transform_fc(transform_type):
    """
    Applies a specific transformation to an image.
    """

    class PadToSquare:
        def __call__(self, image):
            return pad_to_square(image)

    def pad_to_square(image: torch.Tensor) -> torch.Tensor:
        """
        Pads a tensor image [C, H, W] to make it square.
        """
        _, h, w = image.shape
        c = max(h, w)
        padded = torch.zeros(
            (image.shape[0], c, c), dtype=image.dtype, device=image.device
        )
        y_offset = (c - h) // 2
        x_offset = (c - w) // 2
        padded[:, y_offset : y_offset + h, x_offset : x_offset + w] = image
        return padded

    transform_dict = {
        "visualise": v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
        "test": v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),

        "transform_test_padding": v2.Compose(
            [
                v2.ToImage(),
                PadToSquare(),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),


        "transform_data_aug": v2.Compose(
            [
                v2.ToImage(),
                v2.RandomRotation(degrees=5),
                v2.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                v2.ColorJitter(brightness=0.2),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))], p=0.3
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),

        "transform_data_aug_luminosity": v2.Compose(
            [
                v2.ToImage(),
                v2.RandomRotation(degrees=5),
                v2.Resize((224, 224)),
                v2.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        "transform_data_aug_gaussian": v2.Compose(
            [
                v2.ToImage(),
                v2.RandomRotation(degrees=5),
                v2.Resize((224, 224)),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))], p=0.3
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),


        "transform_data_aug_randomerasing_greyscale": v2.Compose(
            [
                v2.ToImage(),
                v2.RandomRotation(degrees=5),
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomErasing(p=0.8, scale=(0.2, 0.3), ratio=(1.0, 3.0), value=0),
                v2.RandomGrayscale(p=0.5),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),

        "transform_data_aug_padding": v2.Compose(
            [
                v2.ToImage(),
                v2.RandomRotation(degrees=5),
                PadToSquare(),
                v2.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                v2.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.02),
                v2.RandomErasing(p=0.8, scale=(0.2, 0.3), ratio=(1.0, 3.0), value=0),
                v2.RandomGrayscale(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.4274, 0.4168, 0.3837], std=[0.1945, 0.1923, 0.1997]
                ),
            ]
        ),
    }

    if transform_type not in transform_dict:
        available = ", ".join(transform_dict.keys())
        raise ValueError(
            f"Unknown transformation: '{transform_type}'. "
            f"Available transformations are: {available}"
        )
    transform = transform_dict.get(transform_type, transform_dict["vi"])

    return transform
