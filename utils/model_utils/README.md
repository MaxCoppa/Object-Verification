# Model Utils  

The model_utils module provides reusable components that support the training and evaluation of Siamese networks for object verification.  
It centralizes functionality for defining losses, computing similarity measures, generating predictions, evaluating performance, and visualizing results.  

---

## Components  

- **losses** - distance-based objectives for training verification models.  
- **similarity** - embedding normalization and similarity computation functions.  
- **predictions** - strategies for generating predictions from similarity scores.  
- **metrics** - evaluation metrics, statistical analyses, and verification measures.  
- **plot_metrics** - visualization utilities for model performance.  

---

## Role in the Pipeline  

- Supplies core objectives and metrics used during training and evaluation.  
- Standardizes similarity and prediction functions to ensure consistency across experiments.  
- Provides interpretable evaluation through quantitative metrics and visual analysis.  
- Designed for extensibility: new losses, similarity functions, and data augmentation strategies can be added without breaking existing workflows.  
- Integrates seamlessly with object_verif_models and dataset_preloader.  

By consolidating these utilities, model_utils ensures that training objectives, evaluation criteria, and reporting tools remain consistent, reproducible, and easily extendable across different experiments.  
