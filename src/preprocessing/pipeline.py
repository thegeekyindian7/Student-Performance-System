from pathlib import Path
import yaml
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class PreprocessingError(Exception):
    pass


def load_preprocessing_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise PreprocessingError(f"Preprocessing config not found: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise PreprocessingError(f"Failed to load preprocessing config: {e}") from e

    if not isinstance(config, dict):
        raise PreprocessingError("Preprocessing config is not a valid YAML mapping")

    return config


def build_preprocessing_pipeline(config: dict) -> ColumnTransformer:
    
    try:
        numerical_features = config["features"]["numerical"]
        categorical_features = config["features"]["categorical"]
    except KeyError as e:
        raise PreprocessingError(f"Missing preprocessing config key: {e}") from e

    
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=config["missing_values"]["numerical"])),
            ("scaler", StandardScaler()),
        ]
    )

    
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=config["missing_values"]["categorical"])),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def fit_transform(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config_path: str,
):
    
    config = load_preprocessing_config(config_path)
    pipeline = build_preprocessing_pipeline(config)

    try:
        X_train_processed = pipeline.fit_transform(X_train)
        X_test_processed = pipeline.transform(X_test)
    except Exception as e:
        raise PreprocessingError(f"Preprocessing failed: {e}") from e

    return X_train_processed, X_test_processed, pipeline
