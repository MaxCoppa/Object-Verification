import numpy as np
from sklearn.metrics import (
    roc_curve,
)
import matplotlib.pyplot as plt


def plot_roc_curve(y_true, y_scores):

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot(fpr, fpr, linestyle="--", color="gray", label="Random guess")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


def plot_score_distributions(y_true, y_scores, threshold=0.5):

    mask_true = y_true == 1

    frr = (y_scores[mask_true] < threshold).sum() / (mask_true).sum() * 100
    far = (y_scores[~mask_true] > threshold).sum() / (~mask_true).sum() * 100

    plt.plot(np.sort(y_scores[mask_true]), label="Positive class")
    plt.plot(np.sort(y_scores[~mask_true]), label="Negative class")
    plt.axhline(
        threshold,
        color="green",
        linestyle="--",
        label=(f"Threshold = {threshold:.2}\nFRR: {frr:.2f}% and FAR: {far:.2f}% "),
    )

    plt.ylabel("Score")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.show()
