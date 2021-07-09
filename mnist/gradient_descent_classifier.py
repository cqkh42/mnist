import torch
import torch.nn.functional as F


class GradientDescentClassifier:
    def __init__(self, lr=1, epochs=200, seed=None):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.seed = seed

    def fit(self, X, y):
        self.initialise(X)
        for i in range(self.epochs):
            self.epoch(X, y)
        return self

    def initialise(self, X):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.weights = torch.randn(X.shape[1]).requires_grad_()
        self.bias = torch.randn(1).requires_grad_()

    def proba(self, X):
        return ((X @ self.weights) + self.bias).sigmoid()


    def epoch(self, X, y):
        epoch_loss = self.loss(X, y)
        epoch_loss.backward()
        self.weights.data -= self.weights.grad.data * self.lr
        self.bias.data -= self.bias.grad.data * self.lr
        self.weights.grad = None
        self.bias.grad = None
        # print(epoch_loss, self.score(X, y))

    def loss(self, X, y_true):
        probs = self.proba(X)
        return torch.where(y_true == 1, 1-probs, probs).mean()

    def predict(self, X):
        probs = self.proba(X)
        return probs.round()

    def score(self, X, y_true):
        preds = self.predict(X)
        return (preds == y_true).float().mean()
