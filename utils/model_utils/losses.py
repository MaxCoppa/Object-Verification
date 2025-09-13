import torch
import torch.nn as nn
import torch.nn.functional as F

from .similarity import (
    euclidean_distance,
    cosine_similarity_distance,
)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss with L2-normalized embeddings.
    Takes embeddings of two samples and a target label == 1 if samples are from the same class, and label == 0 otherwise.
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6  # Small value to avoid division by zero or small distances

    def forward(self, output1, output2, target, size_average=True):

        # Compute squared Euclidean distance between the normalized embeddings
        distances = euclidean_distance(
            embedding1=output1, embedding2=output2, eps=self.eps
        )

        # Contrastive loss calculation
        loss_similar = target.float() * distances.pow(
            2
        )  # If same class, minimize distance
        loss_dissimilar = (1 - target.float()) * F.relu(self.margin - distances).clamp(
            min=self.eps
        ).pow(2)

        losses = 0.5 * (loss_similar + loss_dissimilar)  # Combine both parts

        return losses.mean() if size_average else losses.sum()
