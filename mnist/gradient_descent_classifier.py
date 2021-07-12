import torch

from base_classifier import BaseClassifier
from gradient_descent_mixin import GradientDescentMixin


class GradientDescentClassifier(GradientDescentMixin, BaseClassifier):
    def __init__(self, lr=1, epochs=200, seed=None):
        super().__init__(lr, epochs, seed)
        self.labels = []

    def fit(self, X, y):
        self.initialise(X)
        y = self.normalise_y(y)
        for i in range(self.epochs):
            self.epoch(X, y)
        return self

    def proba(self, X):
        return self._predict(X).sigmoid()

    def loss(self, X, y_true):
        probs = self.proba(X)
        return torch.where(y_true == 1, 1-probs, probs).mean()

    def predict(self, X) -> torch.tensor:
        probs = self.proba(X)
        label_index = probs.round().int()
        return self.to_labels(label_index)
