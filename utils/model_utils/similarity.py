import torch
import math


def l2_normalize(embedding, eps=1e-6):
    """
    Manually L2-normalize a tensor.
    """
    norm = (embedding.pow(2).sum(1, keepdim=True) + eps).sqrt()

    return embedding / norm


def euclidean_distance(embedding1, embedding2, eps=1e-6, normalize_embeddings=True):
    """
    Compute the Euclidean distance between two embeddings.
    If `normalize_embeddings` is True, the embeddings are L2-normalized before computing the distance.
    """
    if normalize_embeddings:
        embedding1 = l2_normalize(embedding1, eps)
        embedding2 = l2_normalize(embedding2, eps)

    distance = (embedding2 - embedding1).pow(2).sum(1).clamp(min=eps).sqrt()
    return distance


def euclidean_similarity_distance(
    embedding1, embedding2, eps=1e-6, normalize_embeddings=True
):
    """
    Compute the Euclidean Similiarity distance between two embeddings.
    If `normalize_embeddings` is True, the embeddings are L2-normalized before computing the distance.
    """
    if normalize_embeddings:
        embedding1 = l2_normalize(embedding1, eps)
        embedding2 = l2_normalize(embedding2, eps)

    distance = (embedding2 - embedding1).pow(2).sum(1).clamp(min=eps).sqrt()
    euclidean_similarity = 1 - distance / math.sqrt(2)  # torch.sqrt(torch.tensor(2.0))

    return euclidean_similarity


def cosine_similarity_distance(embedding1, embedding2, eps=1e-6):
    """
    Compute the cosine similarity between two manually L2-normalized embeddings.
    """
    embedding1 = l2_normalize(embedding1, eps)
    embedding2 = l2_normalize(embedding2, eps)

    cosine_similarity = (embedding1 * embedding2).sum(1)

    return cosine_similarity
