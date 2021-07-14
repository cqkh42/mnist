import torch.nn.functional as F
from torch.utils.data import DataLoader

from gradient_descent_mixin import GradientDescentMixin


def mse(preds, targets): 
    return F.mse_loss(preds, targets)

def mae(preds, targets):
    return F.l1_loss(preds, targets)


class GradientDescentRegressor(GradientDescentMixin):
    def __init__(self, lr=1, epochs=10_000, loss='mse', seed=None, batch_size=5):
        super().__init__(lr, epochs, seed)
        self.batch_size = batch_size

        if loss == 'mse':
            self.loss_func = mse
        elif loss == 'mae':
            self.loss_func = mae
        else:
            raise NotImplementedError

    def fit(self, X, y):
        self.initialise(X)
        dl = DataLoader(
            list(zip(X, y)),
            batch_size=self.batch_size,
            shuffle=True
        )

        for i in range(self.epochs):
            self.epoch(dl)
        return self

    def loss(self, X, y_true):
        preds = self.predict(X)
        return self.loss_func(preds, y_true)

    def predict(self, X):
        return self._predict(X)

    def score(self, X, y):
        predictions = self.predict(X)
        return self.loss_func(predictions, y)
