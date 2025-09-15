# Object Verification with Siamese Networks

This project implements a **Siamese Neural Network** for object verification using **PyTorch**.  
The model learns to determine whether two images belong to the same object class.  

It provides a complete pipeline including:
- Annotation preparation
- Training
- Evaluation
- Prediction

---

## Features

- Siamese model architecture with configurable backbones (default: **ResNet-50**).
- Preprocessing pipeline for raw annotation files.
- Custom PyTorch dataset loader for training and validation.
- Training and evaluation loops with logging and checkpointing.
- Prediction script for testing trained models on image pairs.
- Central configuration file (`config.py`) for paths and parameters.

---

## Repository Structure

### Submodules
Core building blocks of the project:

```

dataset\_annotation\_preparation   # Functions for cleaning and preparing annotations
dataset\_preloader                # Custom PyTorch Dataset for object verification
image\_general                    # Image preprocessing and transform utilities
training\_evaluation              # Training loops, evaluation routines, and metrics
utils                            # Helper functions (image ops, model utils, etc.)
veri\_models                      # Siamese model definitions and backbone architectures

```

### Scripts
Entry-point scripts to run the pipeline:

```

create\_annotation.py             # Prepare dataset annotations from raw CSVs
train\_model.py                   # Train the Siamese model with chosen backbone and loss
demo\_create\_annotation.py        # Demo: run annotation preparation with sample settings
demo\_train\_model.py              # Demo: run training with toy configuration
demo\_prediction\_img.py           # Demo: inference on sample image pairs
model\_to\_onnx.py                 # Export trained model to ONNX format
eval\_veri\_models.ipynb           # Notebook for evaluating trained models

```

### Other Directories
```

data/                            # Raw and preprocessed datasets
logs/                            # Training logs
pretrained\_model/                # Saved model checkpoints

````

---

## Installation

Clone the repository and install dependencies using the requirements.txt file
It is recommended to use a virtual environment to keep dependencies isolated.  



---

## Configuration

All project paths and constants are defined in [`config.py`](./config.py).
This ensures scripts run without hardcoded paths.

Example:

```python
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent

ANNOTATIONS_PATH = PROJECT_PATH / "data" / "preprocessed_annotations"
RAW_ANNOTATIONS_PATH = PROJECT_PATH / "data" / "raw_annotations"
DATA_PATH = PROJECT_PATH / "data" / "images"

MODELS_PATH = PROJECT_PATH / "pretrained_model"
LOGS_PATH = PROJECT_PATH / "logs"
```

> Update these values if your dataset structure differs.

---

## Usage

### 1. Prepare annotations (`create_annotation.py`)

Preprocess raw annotations into a structured CSV file.
This script also supports train/test splitting and data augmentation.

```bash
python create_annotation.py -l True -r 0.7 -e 2 -a 5
```

**Arguments:**

* `--load/-l`: Load images after annotation preparation (default: `False`)
* `--train_ratio/-r`: Train/test split ratio (default: `0.5`)
* `--n_error/-e`: Allowed preprocessing errors (default: `2`)
* `--n_augmentation/-a`: Augmentations per sample (default: `5`)

---

### 2. Train the model (`train_model.py`)

Trains a Siamese network for object verification.
It supports configurable backbones, loss functions, and augmentation settings.

```bash
python train_model.py -e 10 -b resnet50 -a 100 -l Contrastiveloss -n shark_v1
```

**Arguments:**

* `--epochs/-e`: Number of training epochs (default: `1`)
* `--backbone/-b`: Backbone architecture (default: `resnet50`)
* `--augmentation/-a`: Augmentations per sample (default: `100`)
* `--loss/-l`: Loss function (default: `Contrastiveloss`)
* `--model_name/-n`: Prefix for saved model/logs (default: `shark_v1`)
* `--transform_train/-tt`: Training transform (default: `transform_data_aug`)
* `--transform_val/-tv`: Validation transform (default: `test`)

---

### 3. Evaluate

Evaluate trained models using:

* [`eval_veri_models.ipynb`](./eval_veri_models.ipynb)
* or scripts in `training_evaluation/`

---

### 4. Run predictions (`demo_prediction_img.py`)

Run inference on image pairs with a trained model:

```bash
python demo_prediction_img.py
```

This will:

* Generate image pairs
* Perform inference
* Output similarity predictions

---

## Results

(Add training curves, validation metrics, and example predictions here.)

Example:

* Accuracy and loss curves
* Validation metrics (precision, recall, F1-score)
* Example positive and negative predictions

