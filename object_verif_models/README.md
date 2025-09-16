# Verification Models


The `object_verif_models` module provides the core architectures for **verification tasks**.  
It includes:

- **ObjectVeriSiamese**: A Siamese network built on top of various pretrained backbones (ResNet, EfficientNet, ViT, MobileNet).  
- **ModelEnsembler**: An ensemble wrapper to combine multiple Siamese models for more stable and accurate verification.  
- **backbone/**: Folder where pretrained backbone weights must be placed.  

These models are designed for **biometric verification, object verification, and other pairwise similarity tasks**.

---

## Components

### 1. ObjectVeriSiamese

- Flexible architecture supporting multiple backbones:
  - ResNet-50  
  - EfficientNet-V2-S  
  - Vision Transformer (ViT-B/16)  
  - MobileNet-V3-Small  
- Loads pretrained weights from the `backbone/` directory.  
- Projects backbone features into a **128-D embedding space**.  
- Provides a `.predict(image1, image2)` method that outputs predictions and similarity scores.  

---

### 2. ModelEnsembler

- Combines multiple trained Siamese models.  
- Aggregates their similarity scores and averages them.  
- Produces a final binary decision (threshold at 0.5).  
- More robust than single-model verification.  

---

## Backbone Weights

The `backbone/` folder must contain pretrained weights for the supported models.  
Weights can be obtained directly from the official [PyTorch model zoo](https://pytorch.org/vision/stable/models.html).  

Here are the required weights:  

- ResNet-50 → [resnet50-11ad3fa6.pth](https://download.pytorch.org/models/resnet50-11ad3fa6.pth)  
- EfficientNet-V2-S → [efficientnet_v2_s-dd5fe13b.pth](https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth)  
- ViT-B/16 → [vit_b_16-c867db91.pth](https://download.pytorch.org/models/vit_b_16-c867db91.pth)  
- MobileNet-V3-Small → [mobilenet_v3_small-047dcff4.pth](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth)  

Expected folder structure:  

```

veri\_models/
│
├── backbone/
│   ├── resnet50-11ad3fa6.pth
│   ├── efficientnet\_v2\_s-dd5fe13b.pth
│   ├── vit\_b\_16-c867db91.pth
│   ├── mobilenet\_v3\_small-047dcff4.pth
│
├── veri\_models.py        # Contains ObjectVeriSiamese
├── ensemble.py           # Contains ModelEnsembler

```

---

## Features

- **Plug-and-play backbones**.  
- **Frozen or fine-tunable backbone training**.  
- **128-D embedding space** for verification.  
- **Single-model or ensemble evaluation**.  
- Fully compatible with the training and evaluation pipelines.  

---

## Extending with New Backbones

It is possible to extend `ObjectVeriSiamese` with additional backbones by updating its backbone dictionary and adapting the feature extraction logic accordingly.  