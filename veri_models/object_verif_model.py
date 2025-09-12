import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import model_utils

import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


# Definition of the Siamese Network model
class ObjectVeriSiamese(nn.Module):
    def __init__(self, backbone="resnet50", freeze_backbone=True, fc_layer=True):
        super(ObjectVeriSiamese, self).__init__()

        dict_backbone = {
            "efficientnet_v2_s": {
                "model": models.efficientnet_v2_s(),
                "weights_path": "veri_models/backbone/efficientnet_v2_s-dd5fe13b.pth",
                "outuput_shape": 1280,
            },
            "resnet50": {
                "model": models.resnet50(),
                "weights_path": "/veri_models/backbone/resnet50-11ad3fa6.pth",
                "outuput_shape": 2048,
            },
            "vit16": {
                "model": models.vit_b_16(),
                "weights_path": "veri_models/backbone/vit_b_16-c867db91.pth",
                "outuput_shape": 768,
            },
        }
        # Load backbone
        model = dict_backbone[backbone]["model"]
        weights_path = dict_backbone[backbone]["weights_path"]
        model.load_state_dict(
            torch.load(
                weights_path,
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
        )

        # Remove the last FC layer and keep only the feature extractor
        self.backbone = torch.nn.Sequential(*list(model.children())[:-1])

        if backbone in ["vit16", "vit_b_32", "vit_h_14"]:
            model.heads = torch.nn.Identity()
            self.backbone = model

        if backbone in [
            "mobilenet_v2",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "efficientnet_v2_s",
            "efficientnet_v2_m",
            "efficientnet_v2_l",
        ]:
            model.classifier = torch.nn.Identity()
            self.backbone = model

        # Freeze Backbone
        if freeze_backbone:

            self.backbone.eval()

            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add a new Fully Connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(dict_backbone[backbone]["outuput_shape"], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        # Send to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)
        self.fc.to(self.device)
        self.fc_layer = fc_layer

    def forward_once(self, image):
        features = self.backbone(image).squeeze()
        if features.ndim == 1:
            features = features.unsqueeze(0)

        if not self.fc_layer:
            return features

        output = self.fc(features)

        return output

    def forward(self, image1, image2):

        feat1 = self.forward_once(image1)
        feat2 = self.forward_once(image2)

        return feat1, feat2

    def predict(self, image1, image2):
        """
        Returns the probability that the images are similar.
        """

        feat1, feat2 = self.forward(image1, image2)

        preds, similarity_score = model_utils.predict_distance(feat1, feat2)

        return preds, similarity_score
