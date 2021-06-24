import fastai.vision.all as vision
import torch

class MeanClassifier:
    def __init__(self, metric='mae'):
        self.means = {}
        self.metric = metric

    def fit(self, X, y):
        for cls in y.unique():
            self.means[cls.item()] = X[y==cls.item()].mean(0, keepdim=True)
        return self

    def mae(self, X, cls):
        a = (X - self.means[cls]).abs().mean((-1, -2))
        return a

    def mse(self, X, cls):
        return (X - self.means[cls]).pow(2).mean((-1, -2)).sqrt()

    def predict(self, X):
        if self.metric == 'mae':
            metric = self.mae
        elif self.metric == 'mse':
            metric = self.mse
        a = torch.stack([metric(X, i) for i in self.means])
        return torch.argmin(a, axis=0)

    def score(self, X, y, metric='accuracy'):
        predictions = self.predict(X)
        if metric == 'accuracy':
            return (predictions == y).float().mean()

    def show_means(self):
        for mean in self.means:
            vision.show_image(self.means[mean])
