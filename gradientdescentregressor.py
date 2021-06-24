import torch


def mse(preds, targets): 
    return (preds-targets).pow(2).mean()

def mae(preds, targets):
    return (preds-targets).abs().mean()

class GradientDescentRegressor:
    def __init__(self, lr=0.01, epochs=100, loss='mse'):
        self.lr = lr
        self.epochs = epochs
        self.params = None

        if loss == 'mse':
            self.loss_func = mse
        elif loss == 'mae':
            self.loss_func = mae
        else:
            raise NotImplementedError

    def fit(self, X, y):
        self.params = torch.randn(X.shape[1]).requires_grad_()

        for i in range(self.epochs):
            loss = self.score(X, y)
            loss.backward()
            self.params.data -= self.params.grad.data * self.lr
            self.params.grad = None
        return self

    def predict(self, X):
        return X @ self.params

    def score(self, X, y):
        predictions = self.predict(X)
        return mse(predictions, y)