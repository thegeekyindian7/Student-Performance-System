from pathlib import Path
import yaml
import pandas as pd


class DataValidationError(Exception):
    pass


def _load_schema(schema_path: str) -> dict:
    path = Path(schema_path)

    if not path.exists():
        raise DataValidationError(f"Schema file not found: {schema_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            schema = yaml.safe_load(f)
    except Exception as e:
        raise DataValidationError(f"Failed to load schema YAML: {e}") from e

    if not isinstance(schema, dict):
        raise DataValidationError("Schema file is not a valid YAML mapping")

    return schema


def validate_dataframe(df: pd.DataFrame, schema_path: str) -> None:

    schema = _load_schema(schema_path)

    dataset_cfg = schema.get("dataset", {})
    columns_cfg = schema.get("columns", {})

    missing_columns = set(columns_cfg.keys()) - set(df.columns)
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    primary_key = dataset_cfg.get("primary_key")
    if primary_key:
        if primary_key not in df.columns:
            raise DataValidationError(
                f"Primary key '{primary_key}' not found in dataset"
            )

        if df[primary_key].isnull().any():
            raise DataValidationError(
                f"Primary key '{primary_key}' contains null values"
            )

        if df[primary_key].duplicated().any():
            raise DataValidationError(
                f"Primary key '{primary_key}' contains duplicate values"
            )
    for column_name, rules in columns_cfg.items():
        series = df[column_name]

        if not rules.get("nullable", True):
            if series.isnull().any():
                raise DataValidationError(
                    f"Column '{column_name}' contains null values but is marked non-nullable"
                )

        expected_type = rules.get("type")
        if expected_type:
            _validate_type(column_name, series, expected_type)

        if "range" in rules:
            min_val, max_val = rules["range"]
            if series.dropna().lt(min_val).any() or series.dropna().gt(max_val).any():
                raise DataValidationError(
                    f"Column '{column_name}' contains values outside range [{min_val}, {max_val}]"
                )

        if "allowed_values" in rules:
            allowed = set(rules["allowed_values"])
            invalid = set(series.dropna().unique()) - allowed
            if invalid:
                raise DataValidationError(
                    f"Column '{column_name}' contains invalid values: {sorted(invalid)}"
                )


def _validate_type(column_name: str, series: pd.Series, expected_type: str) -> None:
    if expected_type == "integer":
        if not pd.api.types.is_integer_dtype(series):
            raise DataValidationError(
                f"Column '{column_name}' is not of integer type"
            )

    elif expected_type == "float":
        if not pd.api.types.is_float_dtype(series) and not pd.api.types.is_integer_dtype(series):
            raise DataValidationError(
                f"Column '{column_name}' is not of numeric type"
            )

    elif expected_type == "category":

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            raise DataValidationError(
                f"Column '{column_name}' is not categorical"
            )

    elif expected_type == "string":
        if not pd.api.types.is_object_dtype(series):
            raise DataValidationError(
                f"Column '{column_name}' is not of string type"
            )

    else:
        raise DataValidationError(
            f"Unsupported expected type '{expected_type}' for column '{column_name}'"
        )
