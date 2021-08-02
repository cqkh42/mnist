import numpy as np
import torch
from torch.nn import functional as F

from .learner import Learner
from .loss import sigmoid_loss
from .optimiser import SGD


def accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean().item()


def r_2(y_pred, y_true):
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return (1 - (u/v)).item()


class SGDClassifier(Learner):
    def __init__(self, lr, epochs, loss=sigmoid_loss, model=None):
        super().__init__(
            lr=lr, epochs=epochs, optimiser=SGD, model=model, loss=loss)

    def fit(self, dl):
        model_dims = dl.dataset[0][0].shape[0]
        # instantiate model
        self.model = torch.nn.Linear(model_dims, 1)
        self.load_optimiser()
        self._fit(dl)

    def score(self, dl):
        batch_scores = [accuracy(self.predict(X), y) for X, y in dl]
        return np.mean(batch_scores)


class SGDRegressor(Learner):
    def __init__(self, lr, epochs, loss=F.mse_loss, model=None):
        super().__init__(
            lr=lr, epochs=epochs, optimiser=SGD, model=model, loss=loss)

    def score(self, dl):
        batch_scores = [r_2(self.predict(X), y) for X, y in dl]
        return np.mean(batch_scores)

    def fit(self, dl):
        model_dims = dl.dataset[0][0].shape[0]
        # instantiate model
        self.model = torch.nn.Linear(model_dims, 1)
        self.load_optimiser()
        self._fit(dl)
