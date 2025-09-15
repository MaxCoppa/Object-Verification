# Object Verification with Siamese Networks

This project implements a **Siamese Neural Network** for object verification using **PyTorch**.  

Unlike traditional classifiers that predict a specific label for each image, a Siamese Network learns a **similarity function**: it compares two inputs and decides whether they represent the same object.  
This approach is especially effective for verification tasks where the number of possible object classes is very large, constantly growing, or even unseen during training.

By learning a shared embedding space, Siamese Networks generalize beyond fixed categories and provide a robust solution for tasks such as:

- **Face verification** (Are these two faces the same person?)  
- **Signature and handwriting matching**  
- **Product deduplication in e-commerce**  
- **Wildlife re-identification** (tracking individual animals across images)  

This repository provides a **complete pipeline** for object verification with Siamese Networks, covering everything from dataset preparation to model deployment.

---

## Repository Structure

### Submodules (core building blocks)
- **dataset annotation preparation**: preprocess raw annotations into structured CSVs, with train/test splits and augmentations.  
- **dataset preloader**: PyTorch `Dataset` classes for loading image pairs and triplets for training and evaluation.  
- **image general**: utilities for image preprocessing, transformations, and augmentation.  
- **training evaluation**: training loops, evaluation routines, logging, and metric computation.  
- **utils**: helper functions for models, images, ONNX export, and metrics.  
- **veri models**: Siamese model definitions and configurable backbones.  

### Other directories
- **data**: raw annotations, preprocessed CSVs, and image datasets.  
- **logs**: training logs and metric reports.  
- **pretrained model**: saved checkpoints of trained networks.  

---

## Scripts

### `create_annotation.py`
Prepare dataset annotations from raw CSV files.

**Arguments**
- `--load / -l`: Load images after annotation preparation (default: `False`)  
- `--train_ratio / -r`: Train/test split ratio (default: `0.5`)  
- `--n_error / -e`: Allowed preprocessing errors (default: `2`)  
- `--n_augmentation / -a`: Number of augmentations per sample (default: `5`)  

**Outputs**
- Preprocessed CSV file saved under `data/preprocessed_annotations/`.  
- If `--load` is enabled, images are loaded and validated directly via a `DataLoader`.  

**Example**
```bash
python create_annotation.py -l True -r 0.7 -e 2 -a 5
```

---

### `train_model.py`
Train a Siamese network for object verification.  
Supports configurable backbones, loss functions, and augmentation strategies.

**Arguments**
- `--epochs / -e`: Number of training epochs (default: `1`)  
- `--backbone / -b`: Backbone architecture (default: `resnet50`)  
- `--augmentation / -a`: Number of augmentations per sample (default: `100`)  
- `--loss / -l`: Loss function name (default: `Contrastiveloss`)  
- `--model_name / -n`: Prefix for saved model and logs (default: `shark_v1`)  
- `--transform_train / -tt`: Transform type for training (default: `transform_data_aug`)  
- `--transform_val / -tv`: Transform type for validation (default: `test`)  

**Outputs**
- Trained model checkpoint saved in `pretrained_model/model_<model_name>.pth`.  
- Training logs saved in `logs/log_<model_name>.log`.  

**Example**
```bash
python train_model.py -e 10 -b resnet50 -a 100 -l Contrastiveloss -n shark_v1
```

---

### `model_to_onnx.py`
Export a trained Siamese model to ONNX format and compare PyTorch vs ONNX inference.

**Arguments**
- `--model_path / -m`: Path to pretrained PyTorch model (default: `model_1.pth`)  
- `--onnx_path / -o`: Path to save/load ONNX model (default: `model_1.onnx`)  
- `--comparaison_torch_onnx / -b`: Run full comparison loop (default: `False`)  

**Outputs**
- ONNX model exported to `pretrained_model/<onnx_file>.onnx`.  
- Optional speed comparison logs (PyTorch vs ONNX) printed to console.  

**Example**
```bash
python model_to_onnx.py -m model_1.pth -o model_1.onnx -b True
```

---

## Demo and Notebooks
These are provided for experimentation and demonstrations:
- **demo create annotation.py**: sample run of annotation preparation.  
- **demo train model.py**: quick training with toy configuration.  
- **demo prediction img.py**: inference on sample image pairs.  
- **eval veri models.ipynb**: notebook for evaluating trained models interactively.  

---

## Installation and Configuration

To get started:

1. Clone the repository to your workspace.  
2. Create and activate a Python virtual environment (conda or venv recommended).  
3. Install the required dependencies listed in `requirements.txt`.  

All paths for datasets, annotations, models, and logs are centralized in `config.py`.  
If your directory structure differs, adjust the values in this file accordingly.

---

## Usage Overview

1. Prepare annotations → `create_annotation.py` (outputs CSV in `data/preprocessed_annotations/`)  
2. Train the model → `train_model.py` (outputs model in `pretrained_model/`, logs in `logs/`)  
3. Export and compare with ONNX → `model_to_onnx.py` (outputs `.onnx` in `pretrained_model/`)  
4. Use demo scripts or notebooks for quick testing.  

---
