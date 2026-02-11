from src.ingestion.loader import load_csv


def test_ingestion_loads_data():
    df, metadata = load_csv(
        csv_path="data/raw/student_performance.csv",
        metadata_dir="artifacts/test_metadata",
        delimiter=";",
    )

    assert not df.empty
    assert "num_rows" in metadata
    assert metadata["num_rows"] > 0
