import torch
from .similarity import euclidean_similarity_distance, cosine_similarity_distance


def predict_distance(embedding1, embedding2, threshold=0.5):

    distance_similarity = euclidean_similarity_distance(
        embedding1, embedding2, eps=1e-6
    )

    preds = (distance_similarity > threshold).long()
    return preds, distance_similarity


def predict_cosine(embedding1, embedding2, threshold=0.75):

    cosine_similarity = cosine_similarity_distance(embedding1, embedding2, eps=1e-6)
    preds = (cosine_similarity > threshold).long()

    return preds, cosine_similarity


def make_prediction(
    model,
    preprocessed_img1,
    preprocessed_img2,
    predict_fc=predict_distance,
    threshold=0.5,
):

    embedding1, embedding2 = model(preprocessed_img1, preprocessed_img2)

    return predict_fc(embedding1, embedding2, threshold)
