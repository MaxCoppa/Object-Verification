import torch
import torch.nn as nn


class ModelEnsembler(nn.Module):
    """
    Ensemble wrapper for multiple Siamese verification models.

    This module aggregates predictions from multiple trained models by
    averaging their similarity scores. It can be used for more robust
    and stable verification compared to a single model.

    Args:
        models (list[nn.Module]): List of trained models that implement
            a `.predict(image1, image2)` method returning (preds, scores).
        device (str): Target device for computation ("cpu" or "cuda").

    Attributes:
        models (list[nn.Module]): Stored models.
        device (str): Device used for evaluation.
    """

    def __init__(self, models, device="cpu"):
        super(ModelEnsembler, self).__init__()
        self.models = models
        self.device = device
        self.to(self.device)
        self.eval()

    def eval(self):
        for model in self.models:
            model.eval()

    def to(self, device):
        self.device = device
        for model in self.models:
            model.to(device)

    def forward(self, image1, image2):
        """
        Forward pass through the ensemble (averages similarity scores).
        """
        scores = []
        for model in self.models:
            _, score = model.predict(image1, image2)
            scores.append(score.unsqueeze(0))
        scores = torch.cat(scores, dim=0)
        avg_score = scores.mean(dim=0)

        return avg_score

    def predict(self, image1, image2):
        """
        Predict similarity using ensemble of models.

        """
        scores = []
        for model in self.models:
            _, score = model.predict(image1, image2)

            scores.append(score.unsqueeze(0))

        scores = torch.cat(scores, dim=0)

        avg_score = scores.mean(dim=0)
        avg_pred = (avg_score > 0.5).long()

        return avg_pred, avg_score
