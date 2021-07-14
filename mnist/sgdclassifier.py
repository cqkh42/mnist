import torch
from torch.utils.data import DataLoader

from base_classifier import BaseClassifier
from sgdmixin import SGDMixin


class SGDClassifier(SGDMixin, BaseClassifier):
    def __init__(self, lr=1, epochs=200, seed=None, batch_size=5):
        super().__init__(lr, epochs, seed)
        super(BaseClassifier).__init__()
        self.batch_size = batch_size

    def fit(self, X, y):
        self.initialise(X)
        y_norm = self.normalise_y(y).reshape((-1,1))
        dl = DataLoader(
            list(zip(X, y_norm)),
            batch_size=self.batch_size,
            shuffle=True
        )
        for i in range(self.epochs):
            self.epoch(dl)
        return self

    def proba(self, X):
        a = self._predict(X).sigmoid()
        return torch.stack([1-a, a], -1)[:,0]

    def loss(self, X, y_true):
        probs = self.proba(X)
        return torch.where(y_true == 1, probs[:, 0], probs[:, 1]).mean()
