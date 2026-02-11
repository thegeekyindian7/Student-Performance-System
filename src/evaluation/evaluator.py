from pathlib import Path
import json
import pickle
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class EvaluationError(Exception):
    pass


def load_model(model_path: str):
    path = Path(model_path)

    if not path.exists():
        raise EvaluationError(f"Model file not found: {model_path}")

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise EvaluationError(f"Failed to load model: {e}") from e

    return model


def evaluate_model(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
):
    
    model = load_model(model_path)

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        raise EvaluationError(f"Prediction failed: {e}") from e

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=["LOW", "MEDIUM", "HIGH"]
        ).tolist(),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).stem
    metrics_path = output_path / f"{model_name}.json"

    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        raise EvaluationError(f"Failed to save metrics: {e}") from e

    return metrics
