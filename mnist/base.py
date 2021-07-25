from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F


class Base(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y_true):
        pass


class BaseClassifier(Base, ABC):
    @abstractmethod
    def proba(self, X) -> torch.tensor:
        pass

    def score(self, X: torch.tensor, y_true: torch.tensor):
        preds = self.predict(X)
        return (preds == y_true).float().mean().item()

    def predict(self, X: torch.tensor):
        probs = self.proba(X)
        return (probs > 0.5).int()


class BaseRegressor(Base, ABC):
    def score(self, X, y):
        predictions = self.predict(X)
        return F.mse_loss(predictions, y)
