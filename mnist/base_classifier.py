import torch


class BaseClassifier:
    def predict(self, X):
        return NotImplementedError

    def accuracy(self, X, y_true):
        preds = self.predict(X)
        return (preds == y_true).float().mean().item()

    def to_labels(self, predictions):
        return torch.tensor([self.labels[val] for val in predictions])

    def normalise_y(self, y):
        self.labels = y.unique().tolist()
        y = torch.tensor([self.labels.index(val) for val in y])
        return y