import fastai.vision.all as vision
import torch

class MeanClassifier:
    def __init__(self, metric='mae'):
        self.means = {}
        self.metric = metric

    def fit(self, X, y):
        cls_1 = X[y==0]
        cls_2 = X[y==1]
        self.means[0] = cls_1.mean(0)
        self.means[1] = cls_2.mean(0)
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
