import torch
import torch.nn as nn
from utils import model_utils

import torchvision.models as models


# Definition of the Siamese Network model
class ObjectVeriSiameseMC(nn.Module):
    def __init__(self, backbone="resnet50", freeze_backbone=True):
        super(ObjectVeriSiameseMC, self).__init__()

        dict_backbone = {
            "resnet50": {
                "model": models.resnet50(),
                "outuput_shape": 2048,
            },
            "efficientnet_v2_s": {
                "model": models.efficientnet_v2_s(),
                "outuput_shape": 1280,
            },
            "vit16": {
                "model": models.vit_b_16(),
                "outuput_shape": 768,
            },
            "mobilenet_v3_small": {
                "model": models.mobilenet_v3_small(),
                "outuput_shape": 576,
            },
        }
        # Load backbone
        model = dict_backbone[backbone]["model"]

        # Remove the last FC layer and keep only the feature extractor
        self.backbone = torch.nn.Sequential(*list(model.children())[:-1])

        if backbone == "vit16":
            model.heads = torch.nn.Identity()
            self.backbone = model

        if backbone in ["efficientnet_v2_s", "mobilenet_v3_small"]:
            model.classifier = torch.nn.Identity()
            self.backbone = model

        # Freeze Backbone
        if freeze_backbone:

            self.backbone.eval()

            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add a new Fully Connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(dict_backbone[backbone]["outuput_shape"], 128),
            nn.ReLU(),
        )

        # Send to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)
        self.fc.to(self.device)

    def forward_once(self, image):
        features = self.backbone(image).squeeze()

        if features.ndim == 1:
            features = features.unsqueeze(0)

        output = self.fc(features)

        return output

    def forward(self, image1, image2):

        feat1 = self.forward_once(image1)
        feat2 = self.forward_once(image2)

        preds = model_utils.euclidean_similarity_distance(feat1, feat2)

        return preds
