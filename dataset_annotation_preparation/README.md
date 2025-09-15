# Dataset Annotation Preparation

This module provides tools to preprocess raw annotation files into a structured format suitable for **Object Verification with Siamese Networks**.

Currently, it supports **CSV-based annotations**(`prepare_annotation`), producing paired or triplet datasets ready for training and evaluation.

Support for **COCO-style JSON annotations** (`prepare_coco_annotation`) is planned for future updates.

---

## Key Features

- Reads raw CSV annotations and links them to image files.  
- Groups images by object ID and camera.  
- Generates **positive and negative image pairs** (`couples`) or **triplets**.  
- Supports data augmentation through random shuffling of pairs.  
- Splits dataset into **train/test sets** with configurable ratios.  
- Verifies label consistency and valid image paths.  

---

## Main Function

### `prepare_annotation(...)`

Prepares the dataset for object verification from CSV annotations.

**Arguments:**
- `raw_annotation_path (str)` – Path to raw annotation CSV.  
- `images_dir (str)` – Directory containing images.  
- `preprocessed_annotation_path (str)` – Path to save processed CSV.  
- `train_ratio (float)` – Proportion of samples in the training set (default `0.8`).  
- `pairing_type (str)` – `"couples"`, `"triplets"`, or `"test"` (default `"couples"`).  
- `n_augmentation (int)` – Number of shuffled pair sets to generate per object (default `1`).  
- `n_error (int)` – Number of negative pairs or triplets per object (default `1`).  

**Returns:**  
A `pandas.DataFrame` containing processed annotations and saves the same CSV to `preprocessed_annotation_path`.

---

## Output Format

The processed dataset is returned as a **DataFrame** and saved as a CSV. Columns include:

- `img_path` – Path to the first image in the pair.  
- `couple_path` – Path to the second image in the pair.  
- `label` – `1` for a positive pair (same object), `0` for a negative pair (different objects).  
- `train` – `1` if the pair is assigned to the training set, `0` if assigned to the test set.  
- `img_object_id` – Unique identifier for the first image’s object.  
- `couple_object_id` – Unique identifier for the second image’s object.  

**Notes:**  
- Triplet pairs include an additional `error_path` and `error_object_id`.  
- All paths are validated to exist on disk. Invalid entries are removed.  
- Positive/negative labels are verified for consistency.  

Example snippet of the output:

| img_path             | couple_path         | label | train | img_object_id | couple_object_id |
|---------------------|------------------|-------|-------|---------------|-----------------|
| data/images/a.jpg    | data/images/b.jpg | 1     | 1     | obj_001       | obj_001         |
| data/images/c.jpg    | data/images/d.jpg | 0     | 1     | obj_002       | obj_005         |

This format is fully compatible with `VeriImageDataset` in `dataset_preloader`.

---

## Example Usage

```python
from dataset_annotation_preparation.annotation_preprocessor import prepare_annotation

df = prepare_annotation(
    raw_annotation_path="data/raw_annotations/raw_annotations_sharks.csv",
    images_dir="data/images/animals",
    preprocessed_annotation_path="data/preprocessed_annotations/preprocessed_annotations_sharks.csv",
    train_ratio=0.7,
    pairing_type="couples",
    n_augmentation=3,
    n_error=2,
)

print(df.head())
```

---

## Future Updates

* Support for **COCO-style JSON annotations** (`prepare_coco_annotation`).
* Integration of bounding-box cropping for triplets and negative samples.


