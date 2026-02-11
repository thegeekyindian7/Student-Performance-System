from src.labels.derive import derive_labels
import pandas as pd


def test_label_derivation():
    df = pd.DataFrame({"G3": [5, 12, 18]})

    derived = derive_labels(df, "configs/labels.yaml")

    assert set(derived["performance_label"]) == {"LOW", "MEDIUM", "HIGH"}
