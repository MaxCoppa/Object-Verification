import torch
import torch.nn as nn


class ModelEnsembler(nn.Module):
    def __init__(self, models, device="cuda"):
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

        scores = []
        for model in self.models:
            _, score = model.predict(image1, image2)
            scores.append(score.unsqueeze(0))
        scores = torch.cat(scores, dim=0)
        avg_score = scores.mean(dim=0)

        return avg_score

    def predict(self, image1, image2):

        scores = []
        for model in self.models:
            _, score = model.predict(image1, image2)

            scores.append(score.unsqueeze(0))

        scores = torch.cat(scores, dim=0)

        avg_score = scores.mean(dim=0)
        avg_pred = (avg_score > 0.5).long()

        return avg_pred, avg_score
