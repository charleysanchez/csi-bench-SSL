import torch.nn as nn
import torch

def normalize_labels(labels, batch_size, device, criterion):
    if isinstance(labels, tuple):
        labels = labels[0]

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor([int(labels)] * batch_size)

    labels = labels.to(device)

    if isinstance(criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
        labels = labels.float().view(-1, 1)

    return labels
