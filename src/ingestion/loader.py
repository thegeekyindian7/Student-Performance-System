from pathlib import Path
from datetime import datetime
import json
import pandas as pd


class DataIngestionError(Exception):
    pass


def load_csv(csv_path: str, metadata_dir: str, delimiter: str = ","):
    csv_path = Path(csv_path)
    metadata_dir = Path(metadata_dir)

    if not csv_path.exists():
        raise DataIngestionError(f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, sep=delimiter)
    except Exception as e:
        raise DataIngestionError(f"Failed to read CSV: {e}") from e

    if df.empty:
        raise DataIngestionError("Loaded CSV is empty")

    metadata = {
        "source_path": str(csv_path.resolve()),
        "delimiter": delimiter,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": list(df.columns),
        "ingested_at": datetime.utcnow().isoformat(),
    }

    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "ingestion_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return df, metadata
