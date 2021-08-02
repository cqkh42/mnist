from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseClassifier(ABC):
    @abstractmethod
    def proba(self, X) -> torch.tensor:
        pass

    def score(self, dl):
        batch_scores = [(self.predict(X) == y).float().mean() for X, y in dl]
        return np.mean(batch_scores)

    def predict(self, X: torch.tensor):
        probs = self.proba(X)
        return (probs > 0.5).int()
