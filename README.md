# Object Verification with Siamese Networks

This project implements a Siamese Neural Network for object verification using PyTorch.  
The model is designed to determine whether two images belong to the same object class.  
It includes scripts for annotation preparation, training, evaluation, and prediction.

---

## Features

- Siamese model architecture with configurable backbone (default: ResNet-50).
- Preprocessing pipeline for raw annotation files.
- Custom dataset loader for training and validation.
- Training and evaluation loops with logging and checkpointing.
- Prediction script for testing the trained model on image pairs.
- Central configuration file (`config.py`) for paths and parameters.

---

## Repository Structure

```

Object-Verification/
│── README.md                      # Project documentation
│── config.py                      # Project configuration (paths, constants)
│── create_annotation.py           # Utility for generating annotations
│── dataset_annotation_preparation # Prepares cleaned annotations
│── dataset_preloader              # Custom PyTorch dataset
│── image_general                  # Image preprocessing and factory
│── logs/                          # Training logs
│── pretrained_model/              # Saved model checkpoints
│── prediction_img.py              # Run prediction on images
│── test_onxx.py                   # Export or test model in ONNX format
│── train_model.py                 # Train a new Siamese model
│── training_evaluation/           # Training and evaluation utilities
│── training_models_private.py     # Private training configuration
│── training_models_val_test.py    # Training with validation/test splits
│── utils/                         # Helper functions (image, model, etc.)
│── veri_models/                   # Siamese model and related architectures
│── eval_veri_models.ipynb         # Notebook for evaluation
│── data/                          # Dataset directory (images, annotations)
│── **pycache**/                   # Python cache

````

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/Object-Verification.git
cd Object-Verification
pip install -r requirements.txt
````

---

## Configuration

All project paths and constants are stored in `config.py`.
This ensures the code runs without hardcoded absolute paths.

Example from `config.py`:

```python
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent

ANNOTATIONS_PATH = PROJECT_PATH / "data" / "preprocessed_annotations"
RAW_ANNOTATIONS_PATH = PROJECT_PATH / "data" / "raw_annotations"
DATA_PATH = PROJECT_PATH / "data" / "images"

MODELS_PATH = PROJECT_PATH / "pretrained_model"
LOGS_PATH = PROJECT_PATH / "logs"
```

Update these values if your dataset structure differs.

---

## Usage

### 1. Prepare annotations

Run the annotation preparation script to preprocess raw annotation files:

```bash
python create_annotation.py
```

### 2. Train the model

Train the Siamese network using:

```bash
python train_model.py
```

During training:

* Logs will be saved to `logs/`
* Model checkpoints will be saved to `pretrained_model/`

### 3. Evaluate

Use the notebook `eval_veri_models.ipynb` or scripts under `training_models_val_test.py` for evaluation.

### 4. Run predictions

Run the prediction script on images:

```bash
python prediction_img.py
```

This will generate image pairs, perform inference with the trained model, and display the results.

---

## Results

(Add your training curves, validation metrics, and example predictions here.)

Example:

* Training accuracy
* Validation loss curves
* Positive and negative pair predictions

---

## Contributing

Contributions are welcome.
For major changes, please open an issue first to discuss what you would like to modify.

---

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.

