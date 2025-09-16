# Data Directory

The `data/` folder contains all dataset-related resources for training and evaluating Siamese Networks in this project.  
It is divided into three main subdirectories:

---

## Folder Structure

### `images/`
- Contains the raw image data.  
- Images are sourced from the **ImageNet dataset** ([Deng et al., 2009](https://image-net.org/)).  
- For this project, a small demonstration subset is used, consisting of **four animal classes**:  
  - Two different bird species  
  - Two different shark species  
- Each species has exactly **two distinct images**, used for generating verification pairs.

---

### `raw_annotations/`
- Contains the initial **annotation files** describing the dataset in a structured format.  
- These raw files define which images belong to which species/class.  
- They are the input for the preprocessing pipeline.  
- Example (conceptual):  

  ```
  id/image, Camera ID
  bird_A/img1, 0
  bird_A/img2, 0
  shark_B/img1, 1
  shark_B/img2, 1
  ```

---

### `preprocessed_annotations/`
- Contains **processed CSV files** generated from the raw annotations.  
- These files are **private** and not shared in the repository, but they follow a consistent structure.  
- The preprocessing pipeline creates **verification pairs**:
  - **Positive pairs**: two images of the *same object* (e.g., bird_A/img1 and bird_A/img2).  
  - **Negative pairs**: two images of *different objects within the same family* (e.g., bird_A/img1 and bird_B/img2).  
- Expected CSV format (conceptual):  

  ```
  img_path,couple_path,label,train
  data/images/bird_A/img1.jpg,data/images/bird_A/img2.jpg,1,1
  data/images/bird_A/img1.jpg,data/images/bird_B/img2.jpg,0,1
  data/images/shark_A/img1.jpg,data/images/shark_B/img2.jpg,0,0
  ```

  - `img_path`: path to the first image  
  - `couple_path`: path to the paired image  
  - `label`: `1` for positive pairs, `0` for negative pairs  
  - `train`: `1` if the pair is part of the training split, `0` if test/validation  

---

## Workflow

1. **Raw images** (`images/`) are collected from ImageNet.  
2. **Raw annotations** (`raw_annotations/`) map images to species/classes.  
3. **Preprocessing** creates structured CSVs in `preprocessed_annotations/`, which define positive and negative pairs.  
4. These preprocessed files are loaded by the **`dataset_preloader`** to build PyTorch datasets for training and evaluation.

---

## Notes

- The images in this directory are used only for demonstration and testing purposes.  
- The actual preprocessed annotation CSVs are private, but the provided pipeline can generate them from raw annotations.  
- This structure generalizes to larger datasets.  

