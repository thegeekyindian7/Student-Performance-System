from pathlib import Path
import yaml
import pandas as pd


class FeatureEngineeringError(Exception):
    pass


def load_feature_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FeatureEngineeringError(f"Feature config not found: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise FeatureEngineeringError(f"Failed to load feature config: {e}") from e

    if not isinstance(config, dict):
        raise FeatureEngineeringError("Feature config is not a valid YAML mapping")

    return config


def build_features(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    
    config = load_feature_config(config_path)
    feature_defs = config.get("features")

    if not feature_defs:
        raise FeatureEngineeringError("No features defined in feature config")

    features = {}

    for feature in feature_defs:
        name = feature.get("name")
        ftype = feature.get("type")

        if not name or not ftype:
            raise FeatureEngineeringError("Feature definition missing 'name' or 'type'")

    
        if ftype == "direct":
            source = feature.get("source_column")
            if source not in df.columns:
                raise FeatureEngineeringError(
                    f"Source column '{source}' not found for feature '{name}'"
                )
            features[name] = df[source]

    
        elif ftype == "aggregate":
            cols = feature.get("source_columns")
            method = feature.get("method")

            if not cols or not method:
                raise FeatureEngineeringError(
                    f"Aggregate feature '{name}' missing columns or method"
                )

            for col in cols:
                if col not in df.columns:
                    raise FeatureEngineeringError(
                        f"Source column '{col}' not found for feature '{name}'"
                    )

            if method == "mean":
                features[name] = df[cols].mean(axis=1)
            elif method == "min":
                features[name] = df[cols].min(axis=1)
            elif method == "max":
                features[name] = df[cols].max(axis=1)
            elif method == "variance":
                features[name] = df[cols].var(axis=1)
            else:
                raise FeatureEngineeringError(
                    f"Unsupported aggregation method '{method}' for feature '{name}'"
                )

        else:
            raise FeatureEngineeringError(
                f"Unsupported feature type '{ftype}' for feature '{name}'"
            )

    return pd.DataFrame(features)
