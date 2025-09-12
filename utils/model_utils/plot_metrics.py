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


def plot_frr_far(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
    frr = (1 - tpr) * 100
    far = fpr * 100

    plt.plot(frr, far)

    # Lignes verticales pour FRR à 1%, 5%, 10%
    for x_val in [1, 5, 10]:
        plt.axvline(x=x_val, color="gray", linestyle="--", linewidth=1)
        plt.text(
            x=x_val,
            y=1.1,
            s=f"{x_val}%",
            va="bottom",
            ha="left",
            fontsize=9,
            color="gray",
            rotation=90,
        )

    # Échelle log pour les deux axes
    plt.xscale("log")
    plt.yscale("log")

    # Étiquettes
    plt.xlabel("FRR")
    plt.ylabel("FAR")

    # Limites
    plt.xlim(0.1, 100)  # FRR : de 0.1% à 100%
    plt.ylim(0.1, 100)  # FAR : de 1% à 100%

    # Ticks
    xticks = [0.1, 1, 10, 100]
    yticks = [0.1, 1, 10, 100]
    plt.xticks(xticks, [f"{x:.1f}%" if x < 1 else f"{int(x)}%" for x in xticks])
    plt.yticks(yticks, [f"{y:.1f}%" if y < 1 else f"{int(y)}%" for y in yticks])

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
