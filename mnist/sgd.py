import torch
from torch.nn import functional as F

from learner import Learner
from optimiser import SGD


def sigmoid_loss(preds, y_true):
    proba = preds.sigmoid()
    return torch.where(y_true == 1, 1 - proba, proba).mean()


class SGDClassifier(Learner):
    def __init__(self, lr, epochs, model, loss=sigmoid_loss):
        super().__init__(
            lr=lr, epochs=epochs, optimiser=SGD, model=model,
            loss=loss
        )


class SGDRegressor(Learner):
    def __init__(self, lr, epochs, model, loss=F.mse_loss):
        super().__init__(
            lr=lr, epochs=epochs, optimiser=SGD, model=model,
            loss=loss
        )
