from sklearn.ensemble import RandomForestClassifier
import numpy as np

from .base import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, params: dict):
        self.model = RandomForestClassifier(**params)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_name(self) -> str:
        return "random_forest"
