# Training and Evaluation for Siamese Verification Models

## Overview

This submodule provides functionality for **training, validating, and evaluating Siamese neural networks** in biometric verification and other pairwise similarity tasks.  

It integrates **training routines, validation with best-model checkpointing, and evaluation pipelines** with metrics such as **ROC AUC** and **False Rejection Rate (FRR)**. Logging and plotting utilities ensure transparency and reproducibility of experiments.

---

## Workflow

### Training

- Alternates between **training** and **validation** phases.  
- Tracks **loss** and **ROC AUC** across epochs.  
- Saves the **best model weights** based on validation AUC.  
- Optionally plots **AUC curves**.  

---

### Evaluation

- Runs inference on all pairs in the dataset.  
- Computes **ROC AUC**, **FRR** at different thresholds, and quartile statistics.  
- Outputs detailed per-sample predictions in a `pandas.DataFrame`.  

---

## Metrics

- **ROC AUC (Area Under the ROC Curve):** Separability of genuine vs. impostor pairs.  
- **FRR (False Rejection Rate):** Probability of rejecting a genuine pair at a target **FNR (False Negative Rate)**.  
- **Threshold Analysis:** Provides operating points for different security requirements.  

---

## Features

- **Best-model checkpointing** during training.  
- **Detailed evaluation reports** with per-sample outputs.  
- **Logging utilities** for reproducibility.  
- **Optional plotting** of training/validation performance curves.  
