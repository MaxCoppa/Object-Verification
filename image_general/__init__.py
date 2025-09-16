"""
image_module

This package provides a set of high-level classes to handle annotated image processing,
including single image handling, paired image comparison, and transformation utilities.

Public Interface:
- Image: Represents a single image with associated metadata, preprocessing, and visualization methods.
- CoupleImage: Represents a pair of images used for model-based similarity or classification tasks.
- ImagePreprocessor: Loads annotations and provides transformation/model configs for images.
- ImageFactory: Factory pattern class that creates Image and CoupleImage objects consistently.

Usage Example:
    from image_module import Image, CoupleImage, ImagePreprocessor, ImageFactory
"""

__all__ = [
    "Image",
    "CoupleImage",
    "ImagePreprocessor",
    "ImageFactory",
]

from .image import Image
from .couple import CoupleImage
from .loader import ImagePreprocessor
from .image_builder import ImageFactory


__all__ = [
    "Image",
    "CoupleImage",
    "ImagePreprocessor",
    "ImageFactory",
]

from .image import Image
from .couple import CoupleImage
from .loader import ImagePreprocessor
from .image_builder import ImageFactory
