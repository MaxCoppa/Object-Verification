# Image Utils

The **`image_utils`** submodule provides core utilities for image handling in the object verification pipeline.  
It covers **loading, preprocessing, augmentation, visualization, and path management**, ensuring consistency across training and evaluation.

---

## Components
- **`image_loader.py`** – Loading, cropping, anonymization, padding.  
- **`image_transforms.py`** – Predefined transforms for test, visualisation, and augmentation.  
- **`image_plot.py`** – Visualization of image pairs with labels, predictions, and scores.  

---

## Features
- Multi-format image loading (`jpg`, `png`, etc.) with bit-depth correction.  
- Cropping, anonymization, and padding to square shapes.  
- Transform pipelines: `visualise`, `test`, `transform_data_aug`, and variants.  
- Quick visualization of annotated pairs for inspection.  
- Path utilities for consistent dataset handling.  

---

## Integration
- Compatible with **`dataset_preloader`** and **`dataset_annotation_preparation`**.  
- Serves as backend utilities for **`image_general`** (`Image`, `CoupleImage`).  
- Built on **PyTorch** / **torchvision** for seamless DL workflows.  
