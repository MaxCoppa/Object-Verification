# Image General

The **`image_general`** submodule provides a set of abstractions for managing, preprocessing, and pairing images in an **object verification pipeline**.  
It builds the foundation for preparing annotated images, creating training/evaluation pairs, and running model predictions.

---

## Overview

This submodule introduces four main components:

- **`ImagePreprocessor`**  
  Handles annotation parsing, preprocessing setup, device management, and optional model integration.

- **`Image`**  
  Represents a single annotated image, with functionality for loading, transformation, visualization, and metadata access.

- **`CoupleImage`**  
  Represents a pair of images (e.g., two views of the same object). Enables preparation, prediction with models, and visualization of results.

- **`ImageFactory`**  
  A factory class for consistently creating `Image` and `CoupleImage` instances, ensuring they share the same preprocessing configuration.

---

## Key Features

- **Flexible preprocessing**: Centralized through `ImagePreprocessor`, with support for transforms, cropping, and device selection.  
- **Annotation integration**: Works directly with CSV-based annotations from the `dataset_annotation_preparation` submodule.  
- **Pair and single-image handling**: Unified API for managing both individual images and verification pairs.  
- **Metadata access**: Provides ID, timestamp, and folder information for traceability.  
- **Visualization**: Quick utilities for displaying transformed or paired images.

---

## Typical Workflow

1. Initialize an **`ImagePreprocessor`** with annotations and preprocessing settings.  
2. Use **`ImageFactory`** to create `Image` or `CoupleImage` instances from annotation rows.  
3. Apply transformations and load images for model input.  
4. Use **`CoupleImage`** to run predictions with a verification model.  
5. Access metadata and visualize samples for inspection.  

---

## Integration Notes

- Designed for compatibility with datasets prepared by the **`dataset_preloader`** and **`dataset_annotation_preparation`** submodules.  
- Provides the low-level image handling building blocks used by higher-level training and evaluation modules.  
- Can be extended with additional preprocessing or image comparison strategies.
