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


def euclidean_distance_batches(
    embedding1, embedding2, eps=1e-6, normalize_embeddings=True
):
    """
    Compute the Euclidean Similiarity distance between embeddings in batches.
    """
    if normalize_embeddings:
        embedding1 = l2_normalize(embedding1, eps)
        embedding2 = l2_normalize(embedding2, eps)

    mat = embedding1 @ embedding2.T
    similarity_matrix = (2 - 2 * mat).sqrt()

    positive_pairs_similarity = similarity_matrix.diag()

    positive_labels = torch.ones_like(
        positive_pairs_similarity, dtype=torch.long
    )  # [N], label = 1
    similarity_matrix = similarity_matrix.clone()
    idx = torch.arange(similarity_matrix.size(0))
    similarity_matrix[idx, idx] = float("-inf")
    negative_pairs_similarity = torch.topk(
        similarity_matrix, k=min(2, similarity_matrix.size(1) - 1), dim=1
    ).values.mean(dim=1)

    negative_labels = torch.zeros_like(
        negative_pairs_similarity, dtype=torch.long
    )  # [N], label = 0

    # Concatenate and shuffle
    all_pairs_similarity = torch.cat(
        [positive_pairs_similarity, negative_pairs_similarity], dim=0
    )  # [2N]
    all_labels = torch.cat([positive_labels, negative_labels], dim=0)  # [2N]

    indices = torch.randperm(all_pairs_similarity.size(0))  # shuffle indices

    all_pairs_similarity_shuffled = all_pairs_similarity[indices]
    all_labels_shuffled = all_labels[indices]
    return all_pairs_similarity_shuffled, all_labels_shuffled
