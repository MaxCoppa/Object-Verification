# Model utils

The **`model_utils`** submodule provides utility functions and components that support the training and evaluation of **Siamese networks** for object verification. It includes losses, similarity measures, prediction methods, metrics, and visualization tools.

---

## Components

- **losses.py**  
  - Implements distance-based training objectives.

- **similarity.py**  
  - Provides functions for embedding normalization and similarity computation.

- **predictions.py**  
  - Defines prediction strategies based on similarity scores.

- **metrics.py**  
  - Offers evaluation metrics, statistical analyses, and verification measures.

- **plot_metrics.py**  
  - Contains visualization utilities for model performance.

---

## Integration Notes

This submodule works alongside `veri_models` and `dataset_preloader`, ensuring consistent objectives, predictions, and evaluations across experiments.
