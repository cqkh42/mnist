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
        self.classes_ = []

    def fit(self, dl):
        X, y = zip(*dl.dataset)
        X = torch.stack(X)
        y = torch.tensor(y)
        means = []
        for cls in y.unique():
            self.classes_.append(cls)
            mean = X[y == cls].mean(0)
            means.append(mean)
        self.means = torch.stack(means)
        return self

    def proba(self, X):
        preds = torch.stack([self.loss(X, mean) for mean in self.means], -1)
        return preds.softmax(1)

    def score(self, dl):
        batch_scores = [(self.predict(X) == y).float().mean() for X, y in dl]
        return np.mean(batch_scores)

    def predict(self, X: torch.tensor):
        probs = self.proba(X)
        chosen_class = probs.argmax(1)
        return torch.tensor([self.classes_[index] for index in chosen_class])
