import numpy as np

from .losses import (
    ContrastiveLoss,
    FocalLoss,
    CircleLoss,
    CosineLoss,
    ContrastiveLossBatches,
)


def get_loss_function(loss_name):
    loss_eq = {
        "Contrastiveloss": ContrastiveLoss(margin=1),
        "CircleLoss": CircleLoss(m=0.25, gamma=1),
        "FocalLoss": FocalLoss(alpha=0.25, gamma=2.0),
        "FocalLoss_v2": FocalLoss(alpha=0.25, gamma=1.0),
        "FocalLoss_v3": FocalLoss(alpha=0.4, gamma=2.0),
        "FocalLoss_v5": FocalLoss(alpha=0.4, gamma=10.0),
        "FocalLoss_v4": FocalLoss(alpha=0.4, gamma=1.0),
        "CosineLoss": CosineLoss(margin=0.2),
        "ContrastiveLossBis": ContrastiveLoss(margin=np.sqrt(2)),
        "ContrastiveLossBatches": ContrastiveLossBatches(margin=1),
    }

    if loss_name not in loss_eq:
        raise ValueError(f"Unknown loss name: {loss_name}")
    return loss_eq[loss_name]
