import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import logging
from tempfile import TemporaryDirectory
from utils import model_utils
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils.model_utils.metrics import evaluate_FRR


def ttrain_and_eval_veri_model(
    model,
    criterion,
    optimizer,
    dataloaders,
    num_epochs=25,
    freeze_backbone=True,
    save_path="best_model.pth",
    plot=False,
    log_filename=None,
    log_to_console=True,
    verbose=False,
):
    """
    Trains a Siamese model with both training and validation phases, saving the best model permanently.
    """

    logger = setup_logger(log_filename, log_to_console)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    since = time.time()
    best_auc = 0.0

    # Lists to store accuracies for plotting
    train_auc_history = []
    val_auc_history = []

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        torch.save(
            model.state_dict(), best_model_params_path
        )  # Save initial model state

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch}/{num_epochs - 1}")
            logger.info("-" * 10)

            # Training and validation loop for each epoch
            for phase in ["train", "val", "test"]:
                epoch_loss, epoch_auc, epoch_FRR = train_epoch(
                    model,
                    dataloaders[phase],
                    criterion,
                    optimizer,
                    device,
                    phase,
                    freeze_backbone,
                    logger,
                    verbose,
                )

                # Log loss and AUC for each phase
                logger.info(f"{phase} Loss: {epoch_loss:.4f} AUC: {epoch_auc:.4f}")
                if phase in ["val", "test"]:
                    for target_fnr, metrics in epoch_FRR.items():
                        threshold = metrics["Threshold"]
                        fpr_percentage = metrics["FPR"] * 100
                        target_fnr_percentage = metrics["FNR"] * 100

                        logger.info(f"Taux FR Cible: {target_fnr_percentage:.2f}%")
                        logger.info(f"Taux FA Correspondant: {fpr_percentage:.2f}%")

                        logger.info(f"  Threshold: {threshold:.2f}")
                    # Append AUC to history
                    if phase == "train":
                        train_auc_history.append(epoch_auc.item())
                    else:
                        val_auc_history.append(epoch_auc.item())

                # Save the best model based on validation AUC
                if phase == "val" and epoch_auc > best_auc:
                    best_auc = epoch_auc
                    torch.save(model.state_dict(), best_model_params_path)
                logger.info("")

            logger.info("")  # Empty line for readability

        time_elapsed = time.time() - since
        logger.info(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        logger.info(f"Best val AUC: {best_auc:.4f}")

        # Load the best model weights
        model.load_state_dict(torch.load(best_model_params_path))

    # Save the best model permanently
    torch.save(model.state_dict(), save_path)

    # Plotting the AUC curves
    if plot:
        plot_AUC_curve(train_auc_history, val_auc_history, num_epochs)

    return model


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    phase,
    freeze_backbone,
    logger,
    verbose,
):
    """
    Train or evaluate the model for one epoch on the given dataset (train or val).
    """
    if not freeze_backbone:
        model.train() if phase == "train" else model.eval()
    else:
        model.fc.train() if phase == "train" else model.eval()

    running_loss = 0.0
    total_samples = 0

    all_scores = []
    all_labels = []

    for image1, image2, labels in tqdm(dataloader, desc=phase, disable=not verbose):

        image1, image2, labels = (
            image1.to(device),
            image2.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
            output1, output2 = model(image1, image2)
            loss = criterion(output1, output2, labels)
            _, scores = model_utils.predict_distance(output1, output2)

            if phase == "train":
                loss.backward()
                optimizer.step()
        batch_size = image1.size(0)
        running_loss += loss.item()
        total_samples += batch_size
        all_scores.extend(scores.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    epoch_loss = running_loss / len(dataloader)
    epoch_auc = -1
    epoch_FRR = None
    if not (1 - all_labels).all() | (all_labels).all():
        epoch_auc = roc_auc_score(all_labels, all_scores)
        epoch_FRR = evaluate_FRR(all_labels, all_scores, [0.01, 0.05])

    return epoch_loss, epoch_auc, epoch_FRR


def plot_AUC_curve(train_auc_history, val_auc_history, num_epochs):
    """
    Plots the AUC curves for training and validation.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_auc_history, label="Training AUC")
    plt.plot(range(num_epochs), val_auc_history, label="Validation AUC")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.title("Training and Validation AUC Over Epochs")
    plt.legend()
    plt.show()


def setup_logger(log_filename=None, log_to_console=True):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove existing handlers if they exist
    # so we can replace them with new ones.
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    # If a new log file is specified, add new FileHandler
    if log_filename:
        fh = logging.FileHandler(log_filename)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

    # Ensure only one console handler
    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if log_to_console and not has_console_handler:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)

    return logger
