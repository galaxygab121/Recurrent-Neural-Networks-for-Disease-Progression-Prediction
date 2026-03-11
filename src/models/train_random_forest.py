import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_PATH,
    NUMERIC_FEATURES,
    OUTPUT_FIGURES_DIR,
    OUTPUT_METRICS_DIR,
    OUTPUT_MODELS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.features.preprocess import preprocess_pima_data
from src.utils.seed import set_seed


def prepare_data():
    df, _, _ = preprocess_pima_data(DATA_PATH)

    X = df[NUMERIC_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_pipeline():
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_split=6,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return pipeline


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    return y_pred, y_prob, metrics


def save_metrics(metrics: dict):
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS_DIR / "random_forest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

def update_comparison_table(metrics: dict):
    comparison_path = Path("outputs/tables/model_comparison.csv")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([
        {
            "model": "Random Forest",
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        }
    ])

    if comparison_path.exists():
        existing = pd.read_csv(comparison_path)
        existing = existing[existing["model"] != "Random Forest"]
        updated = pd.concat([existing, row], ignore_index=True)
    else:
        updated = row

    updated.to_csv(comparison_path, index=False)

def save_model(model):
    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MODELS_DIR / "random_forest_pipeline.pkl", "wb") as f:
        pickle.dump(model, f)


def plot_confusion(y_test, y_pred):
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
    disp.plot()
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "random_forest_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc(y_test, y_prob):
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="Random Forest")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "random_forest_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_names):
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rf_model = model.named_steps["model"]
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_features, sorted_importances)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "random_forest_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    set_seed(RANDOM_STATE)

    X_train, X_test, y_train, y_test = prepare_data()

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred, y_prob, metrics = evaluate_model(model, X_test, y_test)

    print("Random Forest Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    save_metrics(metrics)
    update_comparison_table(metrics)
    save_model(model)
    plot_confusion(y_test, y_pred)
    plot_roc(y_test, y_prob)
    plot_feature_importance(model, NUMERIC_FEATURES)

    print("\nSaved:")
    print("- metrics to outputs/metrics/random_forest_metrics.json")
    print("- model to outputs/models/random_forest_pipeline.pkl")
    print("- figures to outputs/figures/")


if __name__ == "__main__":
    main()