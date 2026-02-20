from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def predict_proba(self, X):
        raise NotImplementedError(
            "This model does not support probability prediction"
        )