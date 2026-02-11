import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from .base import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, params: dict):
        self.model = XGBClassifier(**params)
        self.label_encoder = LabelEncoder()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # XGBoost requires integer labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def get_name(self) -> str:
        return "xgboost"
