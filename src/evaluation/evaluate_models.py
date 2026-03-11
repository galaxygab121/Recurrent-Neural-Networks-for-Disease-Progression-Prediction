import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUT_FIGURES_DIR, OUTPUT_METRICS_DIR, OUTPUT_TABLES_DIR


def load_metrics():
    metric_files = {
        "Random Forest": OUTPUT_METRICS_DIR / "random_forest_metrics.json",
        "Gradient Boosting": OUTPUT_METRICS_DIR / "gradient_boosting_metrics.json",
        "LSTM": OUTPUT_METRICS_DIR / "lstm_metrics.json",
    }

    rows = []
    for model_name, path in metric_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")

        with open(path, "r") as f:
            metrics = json.load(f)

        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    return pd.DataFrame(rows)


def save_results_table(df: pd.DataFrame):
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_TABLES_DIR / "final_model_comparison.csv", index=False)

    rounded_df = df.copy()
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        rounded_df[col] = rounded_df[col].round(4)

    rounded_df.to_csv(OUTPUT_TABLES_DIR / "final_model_comparison_rounded.csv", index=False)
    return rounded_df


def plot_metric_comparison(df: pd.DataFrame):
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(df["model"], df[metric])
        plt.ylim(0, 1)
        plt.ylabel(metric.upper())
        plt.title(f"Model Comparison: {metric.upper()}")
        plt.tight_layout()
        plt.savefig(OUTPUT_FIGURES_DIR / f"comparison_{metric}.png", dpi=300, bbox_inches="tight")
        plt.show()


def plot_overall_scorecard(df: pd.DataFrame):
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    score_df = df.set_index("model")[metrics]

    plt.figure(figsize=(10, 6))
    for model in score_df.index:
        plt.plot(metrics, score_df.loc[model], marker="o", label=model)

    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Overall Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "overall_model_scorecard.png", dpi=300, bbox_inches="tight")
    plt.show()


def create_rank_summary(df: pd.DataFrame):
    rank_df = df.copy()

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        rank_df[f"{metric}_rank"] = rank_df[metric].rank(ascending=False, method="min")

    rank_df["average_rank"] = rank_df[
        [f"{m}_rank" for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]]
    ].mean(axis=1)

    rank_df = rank_df.sort_values("average_rank")
    rank_df.to_csv(OUTPUT_TABLES_DIR / "model_rank_summary.csv", index=False)
    return rank_df


def main():
    df = load_metrics()
    rounded_df = save_results_table(df)
    plot_metric_comparison(df)
    plot_overall_scorecard(df)
    rank_df = create_rank_summary(df)

    print("Final Model Comparison:")
    print(rounded_df)

    print("\nModel Rank Summary:")
    print(rank_df[["model", "average_rank"]])

    print("\nSaved:")
    print("- outputs/tables/final_model_comparison.csv")
    print("- outputs/tables/final_model_comparison_rounded.csv")
    print("- outputs/tables/model_rank_summary.csv")
    print("- comparison figures in outputs/figures/")


if __name__ == "__main__":
    main()