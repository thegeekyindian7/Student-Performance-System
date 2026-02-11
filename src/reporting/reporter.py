from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ReportingError(Exception):
    pass


LABEL_ORDER = ["LOW", "MEDIUM", "HIGH"]


def load_metrics(metrics_dir: str) -> dict:
    path = Path(metrics_dir)

    if not path.exists():
        raise ReportingError(f"Metrics directory not found: {metrics_dir}")

    metrics_files = list(path.glob("*.json"))
    if not metrics_files:
        raise ReportingError("No metrics JSON files found")

    metrics = {}

    for file in metrics_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                metrics[file.stem] = json.load(f)
        except Exception as e:
            raise ReportingError(f"Failed to load metrics file {file}: {e}") from e

    return metrics


def generate_metrics_csv(metrics: dict, output_path: str):
    rows = []

    for model_name, m in metrics.items():
        rows.append({
            "model": model_name,
            "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return df


def plot_confusion_matrix(cm, model_name: str, output_dir: str):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    output_path = Path(output_dir) / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path)
    plt.close()


def plot_accuracy_comparison(metrics_df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(8, 5))

    ax = sns.barplot(
        data=metrics_df,
        x="model",
        y="accuracy",
    )

    
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.4f}")

    
    for p in ax.patches:
        value = p.get_height()
        ax.annotate(
            f"{value:.4f}",
            (p.get_x() + p.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )

    
    min_acc = metrics_df["accuracy"].min()
    max_acc = metrics_df["accuracy"].max()
    margin = 0.01

    plt.ylim(
        max(0.0, min_acc - margin),
        min(1.0, max_acc + margin),
    )

    plt.title("Model Accuracy Comparison (4 decimal precision)")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.tight_layout()

    output_path = Path(output_dir) / "accuracy_comparison.png"
    plt.savefig(output_path)
    plt.close()



def generate_reports(metrics_dir: str, output_dir: str):
    metrics = load_metrics(metrics_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_df = generate_metrics_csv(
        metrics,
        output_path / "metrics_summary.csv"
    )
    for model_name, m in metrics.items():
        plot_confusion_matrix(
            cm=m["confusion_matrix"],
            model_name=model_name,
            output_dir=output_dir,
        )

    plot_accuracy_comparison(metrics_df, output_dir)
