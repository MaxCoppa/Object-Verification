import torch
import torch.nn as nn
import torch.nn.functional as F

from .similarity import (
    euclidean_distance,
    cosine_similarity_distance,
    euclidean_distance_batches,
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


class ContrastiveLossCosine(nn.Module):
    """
    Contrastive loss with L2-normalized embeddings.
    Takes embeddings of two samples and a target label == 1 if samples are from the same class, and label == 0 otherwise.
    """

    def __init__(self, margin):
        super(ContrastiveLossCosine, self).__init__()
        self.margin = margin
        self.eps = 1e-6  # Small value to avoid division by zero or small distances

    def forward(self, output1, output2, target, size_average=True):

        # Calculate Cosine Similarity between the embeddings
        cosine_similarity = cosine_similarity_distance(
            embedding1=output1, embedding2=output2, eps=self.eps
        )

        cosine_similarity = (cosine_similarity + 1) / 2
        # Contrastive loss calculation
        loss_similar = target.float() * F.relu(self.margin - cosine_similarity).clamp(
            min=self.eps
        ).pow(2)

        loss_dissimilar = (1 - target.float()) * cosine_similarity.pow(2)

        losses = 0.5 * (loss_similar + loss_dissimilar)  # Combine both parts

        return losses.mean() if size_average else losses.sum()


class CosineLoss(nn.Module):
    """
    Contrastive loss with cosine similarity-based loss.
    Takes embeddings of two samples and a target label == 1 if samples are from the same class, and label == 0 otherwise.
    """

    def __init__(self, margin):
        super(CosineLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6  # Small value to avoid division by zero or small distances

    def forward(self, output1, output2, target, size_average=True):

        # Calculate Cosine Similarity between the embeddings
        cosine_similarity = cosine_similarity_distance(
            embedding1=output1, embedding2=output2, eps=self.eps
        )
        # Contrastive loss based on cosine similarity
        loss_similar = (
            1 - cosine_similarity
        ) * target.float()  # Minimize cosine similarity for similar pairs

        loss_dissimilar = F.relu(cosine_similarity - self.margin) * (
            1 - target.float()
        )  # Maximize cosine similarity for dissimilar pairs

        # Combine the losses
        losses = loss_similar + loss_dissimilar

        return losses.mean() if size_average else losses.sum()


class BCEEmbeddingLoss(nn.Module):
    def __init__(self):
        super(BCEEmbeddingLoss, self).__init__()
        self.eps = 1e-6  # Small value to avoid division by zero or very small distances

    def forward(self, output1, output2, target, size_average=True):

        # Calculate L2 distance between the embeddings
        distance = euclidean_distance(
            embedding1=output1, embedding2=output2, eps=self.eps
        )
        similarity_score = 1 - distance / torch.sqrt(torch.tensor(2.0))

        #  scaled_distances = torch.sigmoid(-distances)
        # logits = -scaled_distances  # Smaller distances mean more similar, so we invert the logic with a negative sign
        # Compute the BCE loss between the logits and the target

        loss_similar = (
            -(torch.log(torch.clamp(similarity_score, min=self.eps))) * target.float()
        )

        # Minimize cosine similarity for similar pairs
        loss_dissimilar = -torch.log(
            torch.clamp(1 - similarity_score, min=self.eps)
        ) * (1 - target.float())

        # Combine the losses
        losses = loss_similar + loss_dissimilar

        return losses.mean() if size_average else losses.sum()


class BCEContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(BCEContrastiveLoss, self).__init__()
        self.eps = 1e-6  # Small value to avoid division by zero or very small distances
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):

        # Calculate L2 distance between the embeddings
        distance = euclidean_distance(
            embedding1=output1, embedding2=output2, eps=self.eps
        )

        distance_similar = distance.pow(2)
        distance_dissimilar = F.relu(self.margin - distance).clamp(min=self.eps).pow(2)

        loss_similar = (
            -(torch.log(torch.clamp(1 - distance_similar, min=self.eps)))
            * target.float()
        )

        # Minimize cosine similarity for similar pairs
        loss_dissimilar = -torch.log(
            torch.clamp(1 - distance_dissimilar, min=self.eps)
        ) * (1 - target.float())

        # Combine the losses
        losses = 0.5 * (loss_similar + loss_dissimilar)

        return losses.mean() if size_average else losses.sum()


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
        self.eps = 1e-6  # Small value to avoid division by zero or small distances

    def forward(self, output1, output2, target, size_average=True):

        cosine_similarity = (
            1
            + cosine_similarity_distance(
                embedding1=output1, embedding2=output2, eps=self.eps
            )
        ) / 2

        sp = cosine_similarity[target.bool()]
        sn = cosine_similarity[(1 - target).bool()]
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        logsumexp_p, logsumexp_n = 0, 0
        if (logit_p).numel() > 0:
            logsumexp_p = torch.logsumexp(logit_p, dim=0)
        if (logit_n).numel() > 0:
            logsumexp_n = torch.logsumexp(logit_n, dim=0)

        losses = self.soft_plus(logsumexp_p + logsumexp_n)

        return losses.mean() if size_average else losses.sum()


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification problems.
    """

    def __init__(self, alpha=0.25, gamma=2.0):

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-6  # Small value to avoid division by zero or small distances

    def forward(self, output1, output2, target, size_average=True):
        # Rajoute Classification Couche
        cosine_similarity = cosine_similarity_distance(
            embedding1=output1, embedding2=output2, eps=self.eps
        )

        # There is a problem since the loss doesn't belong to [0,1]
        # Compute focal loss
        pt = cosine_similarity * target.float() + (1 - cosine_similarity) * (
            1 - target.float()
        )  # pt = p if y=1 else 1-p
        pt = pt.clamp(min=self.eps)
        alpha_t = self.alpha * target.float() + (1 - self.alpha) * (1 - target.float())

        losses = -alpha_t * (1 - pt).pow(self.gamma) * pt.log()

        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class ContrastiveLossBatches(nn.Module):
    """
    Contrastive loss applied on batches of embedding pairs with dynamic similarity computation.
    Takes embeddings of two samples and a target label == 1 if samples are from the same class, and label == 0 otherwise.
    """

    def __init__(self, margin):
        super(ContrastiveLossBatches, self).__init__()
        self.margin = margin
        self.eps = 1e-6  # Small value to avoid division by zero or small distances

    def forward(self, output1, output2, target, size_average=True):
        # Validate input: only positive pairs allowed
        if not torch.all(target == 1):
            distances = euclidean_distance(
                embedding1=output1, embedding2=output2, eps=self.eps
            )
        else:
            # Compute squared Euclidean distance between the normalized embeddings
            distances, target = euclidean_distance_batches(
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
