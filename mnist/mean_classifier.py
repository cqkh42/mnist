import torch
import torch.nn.functional as F


from base import BaseClassifier


def mse(X, y):
    return 1/(X - y).pow(2).mean(-1)


def mae(X, y):
    return 1/(X - y).abs().mean(-1)


class MeanClassifier(BaseClassifier):
    def __init__(self, loss='mse'):
        if loss == 'mse':
            self.loss = mse
        elif loss == 'mae':
            self.loss = mae
        else:
            raise NotImplementedError
        super().__init__()

        self.means = []
        self.m = {}

    def fit(self, X, y):
        means = []
        for cls in y.unique():
            mean = X[y[:, 0] == cls].mean(0)
            means.append(mean)
        self.means = torch.stack(means)
        return self

    def proba(self, X):
        preds = torch.stack([self.loss(X, mean) for mean in self.means], -1)
        proba = torch.nn.functional.softmax(preds, -1)[:,1:]
        return proba
