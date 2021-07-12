import torch
import torch.nn.functional as F


from base_classifier import BaseClassifier


def mse(X, y):
    return (X - y).pow(2).mean(-1)


def mae(X, y):
    return (X - y).abs().mean(-1)


class MeanClassifier(BaseClassifier):
    def __init__(self, loss='mse'):
        if loss == 'mse':
            self.loss = mse
        elif loss == 'mae':
            self.loss = mae
        else:
            raise NotImplementedError

        self.means = []
        self.labels = []

    def fit(self, X, y):
        means = []
        y = self.normalise_y(y)
        for cls in y.unique():
            mean = X[y == cls.item()].mean(0)
            means.append(mean)
        self.means = torch.stack(means)
        return self

    def proba(self, X):
        preds = torch.stack([self.loss(X, mean) for mean in self.means], -1)
        proba = torch.nn.functional.softmax(preds, -1)
        return 1 - proba

    def predict(self, X):
        probs = self.proba(X)
        cls = probs.argmax(-1)
        return self.to_labels(cls)
