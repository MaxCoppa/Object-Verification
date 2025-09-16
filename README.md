# Object Verification with Siamese Networks  

**Object verification** is the task of determining whether two images correspond to the same underlying object. Unlike object classification, where the goal is to assign an image to a fixed set of categories, object verification focuses on **pairwise comparison**: given two inputs, decide whether they represent the same instance or not.  

This problem arises in several high-impact domains:  
- **Face verification** – confirming whether two face images belong to the same person.  
- **Signature and handwriting verification** – validating the authenticity of documents.  
- **Product deduplication** – detecting duplicate listings in large-scale e-commerce platforms.  
- **Medical imaging** – comparing scans across time to track disease progression.  

A key challenge in object verification is **scalability**: the number of possible classes can be extremely large, constantly changing, or even unseen during training. Traditional classification models struggle in such settings because they require a fixed label space.  

**Siamese Neural Networks (SNNs)** address this limitation by learning a **similarity function**. Instead of predicting labels, the model maps images into a shared embedding space and evaluates their distance. If the embeddings are close, the inputs are likely the same object; if far apart, they are not. This approach enables:  
- **Generalization to unseen classes**  
- **Robustness to intra-class variation** (e.g., lighting, angle, occlusion)  
- **Efficiency in large-scale retrieval and verification tasks**  

This repository provides a **complete training and deployment pipeline** for object verification with Siamese Networks, covering dataset preparation, model training, ONNX export, and demonstration scripts.  

---

## Repository Structure  

### Core Modules  
- **`dataset_annotation_preparation/`** – preprocess raw annotations into structured CSVs with train/test splits and augmentations.  
- **`dataset_preloader/`** – PyTorch `Dataset` classes for loading image pairs or triplets.  
- **`image_general/`** – utilities for preprocessing, augmentation, and transformations.  
- **`training_evaluation/`** – training loops, evaluation routines, logging, and metric computation.  
- **`utils/`** – helper functions for metrics, model management, and ONNX export.  
- **`veri_models/`** – Siamese architectures with configurable backbones.  

### Supporting Directories  
- **`data/`** – raw annotations, processed CSVs, and datasets.  
- **`logs/`** – training logs and evaluation metrics.  
- **`pretrained_model/`** – saved model checkpoints and exported ONNX files.  

---

## Key Scripts  

### `create_annotation.py`  
Prepares dataset annotations from raw CSVs.  

**Arguments**  
- `--load / -l` – load and validate images after preprocessing (default: `False`)  
- `--train_ratio / -r` – train/test split ratio (default: `0.5`)  
- `--n_error / -e` – allowed preprocessing errors (default: `2`)  
- `--n_augmentation / -a` – number of augmentations per sample (default: `5`)  

**Outputs**  
- Preprocessed annotations stored in `data/preprocessed_annotations/`.  
- Optional validation using PyTorch `DataLoader`.  

**Example**  
```bash
python create_annotation.py -l True -r 0.7 -e 2 -a 5
````

---

### `train_model.py`

Trains a Siamese Network with configurable components.

**Arguments**

* `--epochs / -e` – number of training epochs (default: `1`)
* `--backbone / -b` – CNN backbone architecture (default: `resnet50`)
* `--augmentation / -a` – number of augmentations per sample (default: `100`)
* `--loss / -l` – loss function (default: `ContrastiveLoss`)
* `--model_name / -n` – prefix for saved models/logs (default: `shark_v1`)
* `--transform_train / -tt` – transformation pipeline for training (default: `transform_data_aug`)
* `--transform_val / -tv` – transformation pipeline for validation (default: `test`)

**Outputs**

* Model checkpoint: `pretrained_model/model_<model_name>.pth`
* Training log: `logs/log_<model_name>.log`

**Example**

```bash
python train_model.py -e 10 -b resnet50 -a 100 -l ContrastiveLoss -n shark_v1
```

---

### `model_to_onnx.py`

Exports a trained model to **ONNX** and optionally compares PyTorch vs. ONNX inference.

**Arguments**

* `--model_path / -m` – path to PyTorch checkpoint (default: `model_1.pth`)
* `--onnx_path / -o` – path to save/load ONNX model (default: `model_1.onnx`)
* `--comparaison_torch_onnx / -b` – run inference comparison (default: `False`)

**Outputs**

* ONNX model: `pretrained_model/<onnx_file>.onnx`
* Optional runtime comparison logs.

**Example**

```bash
python model_to_onnx.py -m model_1.pth -o model_1.onnx -b True
```

---

## Demo & Notebooks

Interactive resources for experimentation:

* **`demo_create_annotation.py`** – end-to-end annotation preparation example.
* **`demo_train_model.py`** – quick training demo with modular configs.
* **`demo_prediction_img.py`** – inference on sample image pairs.
* **`eval_veri_models.ipynb`** – notebook for evaluating trained models (e.g., on bird datasets).

---

## Installation & Setup

To use this repository:

1. Clone it locally.
2. Create and activate a Python virtual environment.
3. Install dependencies from `requirements.txt`.

All dataset, model, and log paths are centralized in `config.py`.
Update values if your directory structure differs.

---

## Workflow Overview

1. **Prepare annotations** → `create_annotation.py`
2. **Train the model** → `train_model.py`
3. **Export to ONNX** → `model_to_onnx.py`
4. **Run demos or notebooks** for testing and evaluation.

---

