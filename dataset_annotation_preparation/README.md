# Dataset Annotation Preparation

This submodule provides tools to preprocess raw annotations into a structured format suitable for **object verification with Siamese networks**.  
It converts raw image/object metadata into paired or triplet datasets that can be directly used for training, validation, and evaluation.

---

## Key Features

- Processes raw **CSV annotation files** into a standardized format.  
- Groups images by object ID and camera.  
- Generates **positive and negative pairs** (`couples`) or **triplets**.  
- Enables data augmentation via random shuffling of pairs.  
- Splits data into **train/test subsets** with configurable ratios.  
- Ensures consistency of labels and verifies image paths.  

---

## Main Functionality

Currently, the module provides:

- **`prepare_annotation`**:  
  Converts CSV-based annotations into a structured dataset for verification.  
  Supports both pair-based (`couples`) and triplet-based (`triplets`) sampling.  

Planned extensions include:

- **`prepare_coco_annotation`** (upcoming):  
  Support for COCO-style JSON annotations.  

---

## Output Format

The processed dataset is returned as a structured **CSV** (and a `DataFrame` in memory). Typical columns include:  

- `img_path`: Path to the first image.  
- `couple_path`: Path to the second image.  
- `label`: `1` if both images belong to the same object, `0` otherwise.  
- `train`: `1` if assigned to training, `0` if assigned to testing.  
- `img_object_id`: Identifier of the first image’s object.  
- `couple_object_id`: Identifier of the second image’s object.  

For triplet sampling, additional fields are included, e.g., `error_path` and `error_object_id`.  

All paths are validated, and invalid entries are automatically removed.  

---

## Integration Notes

- The output format is **fully compatible** with the datasets in the `dataset_preloader` submodule.  
- Annotations must reference images stored in the designated dataset folder.  
- Augmentation and error pair/triplet settings allow customization of dataset diversity.  

---

## Future Directions

- Support for **COCO-style JSON annotations**.  
- Built-in handling of bounding box crops for positive and negative samples.  
