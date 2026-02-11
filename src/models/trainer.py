from pathlib import Path
import pickle
import yaml
import numpy as np

from .logistic_regression import LogisticRegressionModel
from .decision_tree import DecisionTreeModel
from .random_forest import RandomForestModel
from .svm import SVMModel
from .xgboost_model import XGBoostModel



class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionModel,
    "decision_tree": DecisionTreeModel,
    "random_forest": RandomForestModel,
    "svm": SVMModel,
    "xgboost": XGBoostModel,
}



def load_model_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise ModelTrainingError(f"Model config not found: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ModelTrainingError(f"Failed to load model config: {e}") from e

    if not isinstance(config, dict):
        raise ModelTrainingError("Model config is not a valid YAML mapping")

    return config


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config_path: str,
    output_dir: str,
):
    config = load_model_config(config_path)
    models_cfg = config.get("models")

    if not models_cfg:
        raise ModelTrainingError("No models defined in model config")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trained_models = {}

    for model_name, model_info in models_cfg.items():
        if model_name not in MODEL_REGISTRY:
            raise ModelTrainingError(f"Unsupported model: {model_name}")

        model_class = MODEL_REGISTRY[model_name]
        params = model_info.get("params", {})

        model = model_class(params)

        try:
            model.train(X_train, y_train)
        except Exception as e:
            raise ModelTrainingError(
                f"Training failed for model '{model_name}': {e}"
            ) from e

        model_path = output_path / f"{model_name}.pkl"
        try:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            raise ModelTrainingError(
                f"Failed to save model '{model_name}': {e}"
            ) from e

        trained_models[model_name] = model_path

    return trained_models
