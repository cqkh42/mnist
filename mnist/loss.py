import torch


def sigmoid_loss(preds, y_true):
    proba = preds.sigmoid()
    return torch.where(y_true == 1, 1 - proba, proba).mean()