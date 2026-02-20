from sklearn.svm import SVC
import numpy as np

from .base import BaseModel


class SVMModel(BaseModel):
    def __init__(self, params: dict):
        self.model = SVC(**params)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_name(self) -> str:
        return "svm"
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
