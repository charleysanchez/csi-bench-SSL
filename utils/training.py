import torch.nn as nn
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_lr(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.0):
    """Warmup learning rate scheduler with cosine annealing. Shared by TaskTrainer and DannTrainer."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def is_binary_criterion(criterion):
    return isinstance(criterion, (nn.BCELoss, nn.BCEWithLogitsLoss))

def predict_from_outputs(outputs, criterion):
    if is_binary_criterion(criterion):
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            return (outputs > 0).long()
        return (outputs > 0.5).long()
    return torch.argmax(outputs, dim=1)
