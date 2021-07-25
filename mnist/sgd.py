import torch
from torch.nn import functional as F

from base import BaseClassifier, BaseRegressor
from optimiser import SGD


def sigmoid_loss(preds, y_true):
    proba = preds.sigmoid()
    return torch.where(y_true == 1, 1 - proba, proba).mean()


class SGDMixin:
    def __init__(self, lr, epochs, optimiser=SGD):
        self.lr = lr
        self.epochs = epochs
        self.linear = None
        self.optimiser_class = optimiser

    def initialise(self, num_params):
        self.linear = torch.nn.Linear(num_params, 1)
        self.optimiser = self.optimiser_class(self.linear.parameters(), self.lr)

    def _calc_gradient(self, X, y):
        epoch_loss = self.loss(X, y)
        epoch_loss.backward()

    def fit_batch(self, X, y):
        self._calc_gradient(X, y)
        self.optimiser.step()
        self.optimiser.zero_grad()

    def fit(self, dl):
        num_params = len(dl.dataset[0][0])
        self.initialise(num_params)
        for i in range(self.epochs):
            for X, y in dl:
                self.fit_batch(X, y)
        return self

    def loss(self, X, y_true):
        preds = self.linear(X)
        return self.loss_func(preds, y_true)


class SGDClassifier(SGDMixin, BaseClassifier):
    def __init__(self, lr=1, epochs=200, optimiser=SGD, loss=sigmoid_loss):
        super().__init__(lr, epochs, optimiser)
        self.loss_func = loss

    def proba(self, X):
        return self.linear(X).sigmoid()


class SGDRegressor(SGDMixin, BaseRegressor):
    def __init__(self, lr=1, epochs=200, loss=F.mse_loss, optimiser=SGD):
        super().__init__(lr, epochs, optimiser)
        self.loss_func = loss

    def predict(self, X):
        return self.linear(X)


