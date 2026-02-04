import torch.nn as nn
import torch

def is_binary_criterion(criterion):
    return isinstance(criterion, (nn.BCELoss, nn.BCEWithLogitsLoss))

def predict_from_outputs(outputs, criterion):
    if is_binary_criterion(criterion):
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            return (outputs > 0).long()
        return (outputs > 0.5).long()
    return torch.argmax(outputs, dim=1)
