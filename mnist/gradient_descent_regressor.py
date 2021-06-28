import torch
import torch.nn.functional as F


def mse(preds, targets): 
    return F.mse_loss(preds, targets).sqrt()

def mae(preds, targets):
    return F.l1_loss(preds, targets)


class GradientDescentRegressor:
    def __init__(self, lr=0.01, epochs=10_000, loss='mse'):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

        if loss == 'mse':
            self.loss_func = mse
        elif loss == 'mae':
            self.loss_func = mae
        else:
            raise NotImplementedError

    def fit(self, X, y):
        self.weights = torch.randn(X.shape[1]).requires_grad_()
        self.bias = torch.randn(1).requires_grad_()

        for i in range(self.epochs):
            loss = self.score(X, y)
            loss.backward()
            self.weights.data -= self.weights.grad.data * self.lr
            self.bias.data -= self.bias.grad.data * self.lr
            self.weights.grad = None
            self.bias.grad = None
        return self

    def predict(self, X):
        return (X @ self.weights) + self.bias

    def score(self, X, y):
        predictions = self.predict(X)
        return self.loss_func(predictions, y)
