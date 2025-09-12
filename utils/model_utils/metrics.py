import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


def calculate_confusion_matrix_metrics(y_true, y_pred):
    """
    Calculate and return the confusion matrix metrics: TNR, FPR, FNR, TPR.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    tnr = tn / (tn + fp)  # True Negative Rate
    fpr = fp / (tn + fp)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate
    tpr = tp / (tp + fn)  # True Positive Rate
    return accuracy, tnr, fpr, fnr, tpr


def calculate_classification_metrics(y_true, y_pred, y_scores):
    """
    Calculate classification metrics: recall, precision, F1 score, AUC, and accuracy.
    """
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)  # Calculate Average Precision (AP)
    return recall, precision, f1, auc, ap


def calculate_quartiles_for_classes(y_true, y_scores):
    """
    Calculate Q0.05, Q1 (25th percentile), median (50th percentile), Q3 (75th percentile), Q0.95 for each class (0 and 1).
    """
    y_scores_array = np.array(y_scores)
    y_true_array = np.array(y_true)

    # Split the y_scores based on y_true (class 0 and class 1)
    scores_class_0 = y_scores_array[y_true_array == 0]
    scores_class_1 = y_scores_array[y_true_array == 1]

    # Calculate the quartiles for class 0
    q0_05_class_0 = np.percentile(scores_class_0, 5)
    q1_class_0 = np.percentile(scores_class_0, 25)
    median_class_0 = np.median(scores_class_0)
    q3_class_0 = np.percentile(scores_class_0, 75)
    q0_95_class_0 = np.percentile(scores_class_0, 95)

    # Calculate the quartiles for class 1
    q0_05_class_1 = np.percentile(scores_class_1, 5)
    q1_class_1 = np.percentile(scores_class_1, 25)
    median_class_1 = np.median(scores_class_1)
    q3_class_1 = np.percentile(scores_class_1, 75)
    q0_95_class_1 = np.percentile(scores_class_1, 95)

    quartiles_results = {
        "Class 0": {
            "Q0.05": q0_05_class_0,
            "Q1": q1_class_0,
            "Median": median_class_0,
            "Q3": q3_class_0,
            "Q0.95": q0_95_class_0,
        },
        "Class 1": {
            "Q0.05": q0_05_class_1,
            "Q1": q1_class_1,
            "Median": median_class_1,
            "Q3": q3_class_1,
            "Q0.95": q0_95_class_1,
        },
    }

    return quartiles_results


def evaluate_FRR(y_true, y_scores, target_fnrs=[0.005, 0.01, 0.05, 0.10, 0.25]):
    """
    Given the true labels (y_true) and predicted scores (y_scores), this function computes
    the False Positive Rate (FPR) and the thresholds that correspond to specified False Negative Rates (FNR).
    """

    # Compute the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate the False Negative Rate (FNR)
    fnr = 1 - tpr

    # Initialize a dictionary to store the results
    results = {}

    # Loop over the desired False Negative Rate (FNR) targets
    for target_fnr in target_fnrs:
        # Find the index where the FNR is closest to the target
        idx = np.argmin(np.abs(fnr - target_fnr))
        # Store the threshold and the corresponding FPR at that index
        results[target_fnr] = {
            "Threshold": thresholds[idx],
            "FPR": fpr[idx],
            "FNR": fnr[idx],
        }

    return results


def evaluate_FAR(y_true, y_scores, target_fprs=[0.005, 0.01, 0.05, 0.10, 0.25]):
    """
    Given the true labels (y_true) and predicted scores (y_scores), this function computes
    the False Negative Rates (FNR) and the thresholds that correspond to specified  False Positive Rate (FPR).
    """

    # Compute the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate the False Negative Rate (FNR)
    fnr = 1 - tpr

    # Initialize a dictionary to store the results
    results = {}

    # Loop over the desired False Positive Rate (FPR) targets
    for target_fpr in target_fprs:
        # Find the index where the FNR is closest to the target
        idx = np.argmin(np.abs(fpr - target_fpr))
        # Store the threshold and the corresponding FPR at that index
        results[target_fpr] = {
            "Threshold": thresholds[idx],
            "FPR": fpr[idx],
            "FNR": fnr[idx],
        }

    return results


def evaluate_model(y_true, y_pred, y_scores, print_results=False):
    """
    Evaluate the model by calculating confusion matrix metrics and classification metrics
    such as recall, precision, F1-score, AUC, Accuracy, and AP. Also calculates quartile-based statistics
    for class 0 and class 1.
    """
    # Calculate confusion matrix metrics
    accuracy, tnr, fpr, fnr, tpr = calculate_confusion_matrix_metrics(y_true, y_pred)

    # Calculate other classification metrics
    recall, precision, f1, auc, ap = calculate_classification_metrics(
        y_true, y_pred, y_scores
    )

    # Calculate quartiles for class 0 and class 1
    quartiles_results = calculate_quartiles_for_classes(y_true, y_scores)

    FRR_results = evaluate_FRR(y_true, y_scores)
    FAR_results = evaluate_FAR(y_true, y_scores)

    # Prepare results as a dictionary
    results = {
        "Accuracy": accuracy,
        "AUC": auc,
        "TNR": tnr,
        "FPR": fpr,
        "FNR": fnr,
        "TPR": tpr,
        "Recall": recall,
        "Precision": precision,
        "F1-Score": f1,
        "Average Precision (AP)": ap,  # Added AP to the results
    }

    if print_results:
        # Print results if print_results is True
        print(
            f"Accuracy: {results['Accuracy']:.4f}\n"
            f"AUC: {results['AUC']:.4f}\n"
            f"TNR: {results['TNR']:.4f}, FPR: {results['FPR']:.4f}, FNR: {results['FNR']:.4f}, TPR: {results['TPR']:.4f}\n"
            f"Recall: {results['Recall']:.4f}\n"
            f"Precision: {results['Precision']:.4f}\n"
            f"F1-Score: {results['F1-Score']:.4f}\n"
            f"Average Precision (AP): {results['Average Precision (AP)']:.4f}"
        )

        # Print quartile results for each class
        for class_label, quartiles in quartiles_results.items():
            print(f"\nQuartiles for {class_label}:")
            for quartile, value in quartiles.items():
                print(f"  {quartile}: {value:.4f}")

        print()
        for target_fnr, metrics in FRR_results.items():
            # Format the values to two significant digits and convert FPR/FNR to percentage
            threshold = metrics["Threshold"]
            fpr_percentage = metrics["FPR"] * 100  # Convert to percentage
            target_fnr_percentage = metrics["FNR"] * 100  # Convert to percentage

            print(f"Taux FR Cible: {target_fnr_percentage:.2f}%")
            print(f"Taux FA Correspondant: {fpr_percentage:.2f}%")

            print(f"  Threshold: {threshold:.2f}")
            print()

        for target_fpr, metrics in FAR_results.items():
            # Format the values to two significant digits and convert FPR/FNR to percentage
            threshold = metrics["Threshold"]
            target_fpr_percentage = metrics["FPR"] * 100  # Convert to percentage
            fnr_percentage = metrics["FNR"] * 100  # Convert to percentage

            print(f"Taux FA Cible: {target_fpr_percentage:.2f}%")
            print(f"Taux FR Correspondant: {fnr_percentage:.2f}%")

            print(f"  Threshold: {threshold:.2f}")
            print()

    return results, quartiles_results, FRR_results
