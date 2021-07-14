import torch


class GradientDescentMixin:
    def __init__(self, lr, epochs, seed):
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def initialise(self, X):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.linear = torch.nn.Linear(X.shape[1], 1)
        self.weights = torch.randn(X.shape[1]).requires_grad_()
        self.bias = torch.randn(1).requires_grad_()

    def fit_batch(self, X, y):
        epoch_loss = self.loss(X, y)
        epoch_loss.backward()
        self.weights.data -= self.weights.grad.data * self.lr
        self.bias.data -= self.bias.grad.data * self.lr
        self.weights.grad = None
        self.bias.grad = None

    def epoch(self, dl):
        for X, y in dl:
            self.fit_batch(X, y)

    def _predict(self, X):
        return (X @ self.weights) + self.bias
