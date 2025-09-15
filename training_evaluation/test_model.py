import torch
import os
import time
import pandas as pd
from utils import model_utils
from tqdm import tqdm

import torch


def test_model(model, dataloader, device="cpu", print_results=True):
    """
    Evaluate a Siamese verification model on a given dataset.

    Args:
        model (torch.nn.Module): The trained Siamese model to evaluate.
        dataloader (DataLoader): DataLoader that yields batches of
            (image1, image2, label, image1_path, image2_path).
        device (str): Device to run evaluation on ("cuda" or "cpu").
        print_results (bool): If True, prints evaluation metrics.

    Returns:
        tuple:
            - results_df (pd.DataFrame): Per-sample evaluation results with predictions and scores.
            - results (dict): Global evaluation metrics (e.g., accuracy, AUC).
            - quartiles_results (dict): Metrics aggregated by score quartiles.
            - FRR_results (dict): False Rejection Rate metrics at selected thresholds.
    """
    start_time = time.time()
    print(f"Started Evaluation at {time.ctime(start_time)}")
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    y_true = []
    y_pred = []
    y_scores = []
    img_paths = []
    img_couples = []

    # Loop over the data in the dataloader
    for image1, image2, label, image1_path, image2_path in tqdm(
        dataloader, desc="Evaluating"
    ):

        image1, image2 = image1.to(device), image2.to(device)

        # Inference, no gradients needed
        with torch.no_grad():
            # output1, output2 = model(image1, image2)
            # preds, similarity_score = model_utils.predict_distance(output1, output2)
            preds, similarity_score = model.predict(image1, image2)

            # Append results to lists
            img_paths.extend(image1_path)
            img_couples.extend(image2_path)
            y_true.extend(label.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_scores.extend(similarity_score.cpu().numpy().tolist())

    print(f"Prediction phase took {time.time() - start_time:.2f} seconds.")

    results_df = pd.DataFrame(
        {
            "img_path": img_paths,
            "couple_path": img_couples,
            "label": y_true,
            "prediction": y_pred,
            "distance": y_scores,
        }
    )
    results, quartiles_results, FRR_results = model_utils.evaluate_model(
        y_true, y_pred, y_scores, print_results=print_results
    )

    # Total evaluation time
    total_time = time.time() - start_time
    print(f"Total evaluation process took {total_time:.2f} seconds.")
    return results_df, results, quartiles_results, FRR_results
