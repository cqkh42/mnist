import numpy as np
import torch
import torch.nn.functional as F


def mse(X, y):
    return 1/(X - y).pow(2).mean(-1)


def mae(X, y):
    return 1/(X - y).abs().mean(-1)


class MeanClassifier:
    def __init__(self, loss):
        self.loss = loss
        self.means = []

    def fit(self, dl):
        X, y = zip(*dl.dataset)
        X = torch.stack(X)
        y = torch.stack(y)
        means = []
        for cls in y.unique():
            mean = X[y[:, 0] == cls].mean(0)
            means.append(mean)
        self.means = torch.stack(means)
        return self

    def proba(self, X):
        preds = torch.stack([self.loss(X, mean) for mean in self.means], -1)
        proba = torch.nn.functional.softmax(preds, -1)[:, 1:]
        return proba

    def score(self, dl):
        batch_scores = [(self.predict(X) == y).float().mean() for X, y in dl]
        return np.mean(batch_scores)

    def predict(self, X: torch.tensor):
        probs = self.proba(X)
        return (probs > 0.5).int()
