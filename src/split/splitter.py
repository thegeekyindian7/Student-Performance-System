from pathlib import Path
from datetime import datetime
import json
import yaml
import pandas as pd

from sklearn.model_selection import train_test_split


class DataSplitError(Exception):
    pass


def load_split_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise DataSplitError(f"Split config not found: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise DataSplitError(f"Failed to load split config: {e}") from e

    if not isinstance(config, dict):
        raise DataSplitError("Split config is not a valid YAML mapping")

    return config


def _class_distribution(y: pd.Series) -> dict:
    return y.value_counts(normalize=False).to_dict()


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config_path: str,
    metadata_dir: str,
):
    
    config = load_split_config(config_path)
    split_cfg = config.get("split")

    if not split_cfg:
        raise DataSplitError("Missing 'split' section in split config")

    try:
        train_ratio = split_cfg["train_ratio"]
        test_ratio = split_cfg["test_ratio"]
        stratify = split_cfg.get("stratify", True)
        random_seed = split_cfg["random_seed"]
    except KeyError as e:
        raise DataSplitError(f"Missing split config key: {e}") from e

    if not abs(train_ratio + test_ratio - 1.0) < 1e-6:
        raise DataSplitError("Train and test ratios must sum to 1.0")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_ratio,
            test_size=test_ratio,
            stratify=y if stratify else None,
            random_state=random_seed,
        )
    except Exception as e:
        raise DataSplitError(f"Data splitting failed: {e}") from e


    metadata = {
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "stratified": stratify,
        "random_seed": random_seed,
        "num_samples": {
            "train": len(y_train),
            "test": len(y_test),
        },
        "class_distribution": {
            "train": _class_distribution(y_train),
            "test": _class_distribution(y_test),
        },
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        metadata_path = Path(metadata_dir)
        metadata_path.mkdir(parents=True, exist_ok=True)
        with open(metadata_path / "split_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        raise DataSplitError(f"Failed to save split metadata: {e}") from e

    return X_train, X_test, y_train, y_test
