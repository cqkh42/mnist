import torch


def mse(X, y):
    return (X - y).pow(2).mean(-1)


def mae(X, y):
    return (X - y).abs().mean(-1)


class MeanClassifier:
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
        for cls in y.unique():
            mean = X[y==cls.item()].mean(0)
            self.labels.append(cls.item())
            means.append(mean)
        self.means = torch.stack(means)

    def predict(self, X):
        preds = torch.stack([self.loss(X, mean) for mean in self.means], -1).argmin(-1)
        preds = torch.tensor([self.labels[i] for i in preds])    
        return preds   

    def score(self, X, y_true, metric='accuracy'):
        if metric != 'accuracy':
            raise NotImplementedError
        preds = self.predict(X)
        return (y_true == preds).float().mean().item()
