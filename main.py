import argparse
from venv import logger
import yaml
import sys
from pathlib import Path
from src.ingestion.loader import load_csv
from src.validation.validator import validate_dataframe
from src.features.builder import build_features
from src.split.splitter import split_data
from src.preprocessing.pipeline import fit_transform
from src.models.trainer import train_models
from src.evaluation.evaluator import evaluate_model
from src.reporting.reporter import generate_reports
from src.labels.derive import derive_labels
from src.utils.logger import setup_logger



ARTIFACTS_DIR = Path("artifacts")
METADATA_DIR = ARTIFACTS_DIR / "metadata"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def load_experiment_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str):
    config = load_experiment_config(config_path)
    logger = setup_logger()
    logger.info("Pipeline started")

    df, _ = load_csv(
        csv_path=config["data"]["raw_csv_path"],
        metadata_dir=METADATA_DIR,
        delimiter=config["data"].get("delimiter", ","),
    )
    logger.info("Step 1: Data ingestion started")

    df = derive_labels(
    df=df,
    config_path=config["labels"]["config_path"],
    )
    logger.info("Step 2: Label derivation completed")
    
    validate_dataframe(
        df=df,
        schema_path=config["validation"]["schema_path"],
    )
    logger.info("Step 3: Schema validation passed")
    
    y = df[config["features"]["label_column"]]
    X = build_features(
        df=df,
        config_path=config["features"]["config_path"],
    )
    logger.info("Step 4: Feature engineering completed")
    
    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y=y,
        config_path=config["split"]["config_path"],
        metadata_dir=METADATA_DIR,
    )
    logger.info("Step 5: Data split completed")
        
    X_train_p, X_test_p, _ = fit_transform(
        X_train=X_train,
        X_test=X_test,
        config_path=config["preprocessing"]["config_path"],
    )
    logger.info("Step 6: Preprocessing completed")
    
    trained_models = train_models(
        X_train=X_train_p,
        y_train=y_train.to_numpy(),
        config_path=config["models"]["config_path"],
        output_dir=MODELS_DIR,
    )
    logger.info("Step 7: Model training completed")
    
    for model_path in trained_models.values():
        evaluate_model(
            model_path=model_path,
            X_test=X_test_p,
            y_test=y_test.to_numpy(),
            output_dir=METRICS_DIR,
        )
       
    generate_reports(
        metrics_dir=METRICS_DIR,
        output_dir=PLOTS_DIR,
    )
    logger.info("Step 8: Evaluation and reporting completed")
    logger.info("Pipeline finished successfully")

def main():
    parser = argparse.ArgumentParser(
        description="Student Performance ML Pipeline"
    )
    parser.add_argument(
        "command",
        choices=[
            "run",
            "ingest",
            "validate",
            "features",
            "split",
            "preprocess",
            "train",
            "evaluate",
            "report",
        ],
        help="Pipeline command to run",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to experiment config",
    )

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(args.config)
    else:
        print(
            f" Stepwise execution requires manual wiring.\n"
            f"Use `run` for full pipeline.\n"
            f"Command '{args.command}' is intentionally restricted."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()