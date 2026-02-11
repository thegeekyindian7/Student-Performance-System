from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all models.
    Enforces a uniform training and prediction interface.
    """

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return model name."""
        pass
