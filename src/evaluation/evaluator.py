from pathlib import Path
import json
import pickle
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

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
    
    
    try:
        y_proba = model.predict_proba(X_test)
    except AttributeError:
        raise EvaluationError(
            "Model does not support probability prediction (predict_proba required for ROC)"
        )
    
    class_labels = ["LOW", "MEDIUM", "HIGH"]
    y_test_bin = label_binarize(y_test, classes=class_labels)

    roc_data = {}

    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        roc_data[label] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": roc_auc,
        }

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
        "roc": roc_data,
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