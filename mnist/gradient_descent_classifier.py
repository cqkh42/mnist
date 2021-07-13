import torch

from base_classifier import BaseClassifier
from gradient_descent_mixin import GradientDescentMixin


class GradientDescentClassifier(GradientDescentMixin, BaseClassifier):
    def __init__(self, lr=1, epochs=200, seed=None):
        super().__init__(lr, epochs, seed)
        super(BaseClassifier).__init__()

    def fit(self, X, y):
        self.initialise(X)
        y = self.normalise_y(y)
        for i in range(self.epochs):
            self.epoch(X, y)
        return self

    def proba(self, X):
        a = self._predict(X).sigmoid()
        return torch.stack([1-a, a], -1)

    def loss(self, X, y_true):
        probs = self.proba(X)
        return torch.where(y_true == 1, probs[:, 0], probs[:, 1]).mean()
