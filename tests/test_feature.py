from src.features.builder import build_features
import pandas as pd


def test_feature_builder_outputs_expected_columns():
    df = pd.DataFrame({
        "G1": [10, 12],
        "G2": [11, 13],
        "studytime": [2, 3],
        "failures": [0, 1],
        "absences": [4, 6],
    })

    X = build_features(df, "configs/features.yaml")

    expected_columns = {
        "avg_internal_score",
        "score_variance",
        "study_time",
        "past_failures",
        "absences",
    }

    assert set(X.columns) == expected_columns
