import numpy as np

from .losses import (
    ContrastiveLoss,
)


def get_loss_function(loss_name):
    loss_eq = {
        "Contrastiveloss": ContrastiveLoss(margin=1),
    }

    if loss_name not in loss_eq:
        raise ValueError(f"Unknown loss name: {loss_name}")
    return loss_eq[loss_name]
