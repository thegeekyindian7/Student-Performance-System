from pathlib import Path
import yaml
import pandas as pd


class LabelDerivationError(Exception):
    pass


def load_label_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise LabelDerivationError(f"Label config not found: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise LabelDerivationError(f"Failed to load label config: {e}") from e

    if not isinstance(config, dict):
        raise LabelDerivationError("Label config is not a valid YAML mapping")

    return config


def derive_labels(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    
    config = load_label_config(config_path)

    score_col = config.get("source_score_column")
    label_col = config.get("label_column")
    bins = config.get("bins")
    labels = config.get("labels")

    if not all([score_col, label_col, bins, labels]):
        raise LabelDerivationError("Incomplete label derivation configuration")

    if score_col not in df.columns:
        raise LabelDerivationError(
            f"Score column '{score_col}' not found in dataset"
        )

    if len(bins) - 1 != len(labels):
        raise LabelDerivationError(
            "Number of labels must be exactly len(bins) - 1"
        )

    derived_df = df.copy()

    try:
        derived_df[label_col] = pd.cut(
            derived_df[score_col],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
    except Exception as e:
        raise LabelDerivationError(f"Label derivation failed: {e}") from e

    if derived_df[label_col].isnull().any():
        raise LabelDerivationError(
            "Label derivation produced null labels; check bin definitions"
        )

    return derived_df
