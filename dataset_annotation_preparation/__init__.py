"""
The aim of this submodule is to create prepare and create dataset annotations
con
"""

__all__ = ["prepare_annotation", "prepare_coco_annotation"]


from .annotation_preprocessor import prepare_annotation
from .coco_annotation_preprocessor import prepare_coco_annotation
