# Dataset Preloader

The **Dataset Preloader** submodule provides custom PyTorch `Dataset` classes for the **Object Verification** pipeline.  
It handles loading of image pairs or triplets from preprocessed annotation files, making them directly usable in **training, validation, and evaluation** of Siamese networks.

---

## VeriImageDataset

This dataset is intended for **training** Siamese networks.  
It supports both **pair-based** and **triplet-based** sampling depending on configuration.

### Arguments

- **annotations_file** (`str`): Path to the CSV file with preprocessed annotations.  
- **correct_pair_ratio** (`float`): Probability of sampling a positive pair in triplet mode. Default: `0.5`.  
- **img_dir** (`str | None`): Optional base directory for image files.  
- **train** (`bool`): Whether to load the training set (`True`) or the test set (`False`).  
- **transform** (`callable | None`): Optional torchvision transform(s) applied to images.  
- **file_type** (`str`): Image format (e.g. `"jpg"`, `"png"`). Default: `"jpg"`.  
- **pairing_type** (`str`): Pairing strategy, `"couples"` for pairs or `"triplets"` for triplets.  
- **crop_type** (`str | None`): Cropping strategy to apply if needed.  

### Returned Item

Each sample has the format:

```

(image\_tensor, couple\_tensor, label)

```

- `image_tensor`: First image tensor.  
- `couple_tensor`: Second image tensor.  
- `label`: `1` if both images belong to the same object, `0` otherwise.  

### Features

- Flexible support for **pairs** and **triplets**.  
- Compatible with custom transforms and crop strategies.  
- Visualization utilities to inspect training samples.  

---

## VeriImageDatasetTest

This dataset is tailored for **evaluation and testing**.  
In addition to images and labels, it also provides file paths for **traceability**.

### Arguments

- **annotations_file** (`str`): Path to the CSV file with preprocessed annotations.  
- **img_dir** (`str | None`): Optional base directory for image files.  
- **transform** (`callable | None`): Optional torchvision transform(s) applied to images.  
- **file_type** (`str`): Image format (e.g. `"jpg"`, `"png"`). Default: `"jpg"`.  
- **crop_type** (`str | None`): Cropping strategy to apply if needed.  

### Returned Item

Each sample has the format:

```

(image\_tensor, couple\_tensor, label, image\_path, couple\_path)

```

- `image_tensor`: First image tensor.  
- `couple_tensor`: Second image tensor.  
- `label`: `1` for a positive pair, `0` for a negative pair.  
- `image_path`: Path to the first image.  
- `couple_path`: Path to the second image.  

### Features

- Provides **traceability** by returning file paths with tensors.  
- Supports cropping and transforms during evaluation.  
- Visualization utilities to inspect evaluation samples.  

---

## Integration Notes

- Both datasets operate on CSV annotation files produced by the **`dataset_annotation_preparation`** submodule.  
- They integrate seamlessly with the **training** and **evaluation** pipelines for Siamese models.  
- Designed for flexibility across pairing strategies, image transforms, and cropping workflows.  
```
