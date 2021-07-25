from abc import ABC, abstractmethod

import torch


class BaseClassifier(ABC):
    @abstractmethod
    def proba(self, X) -> torch.tensor:
        pass

    def score(self, X: torch.tensor, y_true: torch.tensor):
        preds = self.predict(X)
        return (preds == y_true).float().mean().item()

    def predict(self, X: torch.tensor):
        probs = self.proba(X)
        return (probs > 0.5).int()
