import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        logits = self.classifier(last_hidden)
        return logits.squeeze(1)


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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # reshape into [samples, sequence_length, input_size]
    X_train_seq = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_seq = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    return (
        X_train_tensor,
        X_test_tensor,
        y_train_tensor,
        y_test_tensor,
        scaler,
        y_test.values,
    )


def build_dataloaders(X_train, y_train, batch_size=32):
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train_model(model, train_loader, epochs=50, learning_rate=1e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": []}

    model.train()
    for epoch in range(epochs):
        epoch_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_loss:.4f}")

    return history


def evaluate_model(model, X_test_tensor, y_test_true):
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor.to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test_true, preds)),
        "precision": float(precision_score(y_test_true, preds)),
        "recall": float(recall_score(y_test_true, preds)),
        "f1": float(f1_score(y_test_true, preds)),
        "roc_auc": float(roc_auc_score(y_test_true, probs)),
    }

    return preds, probs, metrics


def save_metrics(metrics: dict):
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS_DIR / "lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


def update_comparison_table(metrics: dict):
    comparison_path = Path("outputs/tables/model_comparison.csv")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([
        {
            "model": "LSTM",
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        }
    ])

    if comparison_path.exists():
        existing = pd.read_csv(comparison_path)
        existing = existing[existing["model"] != "LSTM"]
        updated = pd.concat([existing, row], ignore_index=True)
    else:
        updated = row

    updated.to_csv(comparison_path, index=False)


def save_model(model, scaler):
    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": NUMERIC_FEATURES,
        },
        OUTPUT_MODELS_DIR / "lstm_model.pt",
    )

    import pickle
    with open(OUTPUT_MODELS_DIR / "lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


def plot_training_curve(history):
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("LSTM Training Loss")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "lstm_training_loss.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion(y_test, y_pred):
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
    disp.plot()
    plt.title("LSTM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "lstm_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc(y_test, y_prob):
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="LSTM")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LSTM ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES_DIR / "lstm_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    set_seed(RANDOM_STATE)

    (
        X_train_tensor,
        X_test_tensor,
        y_train_tensor,
        y_test_tensor,
        scaler,
        y_test_true,
    ) = prepare_data()

    train_loader = build_dataloaders(X_train_tensor, y_train_tensor, batch_size=32)

    model = TabularLSTM(input_size=1, hidden_size=32, num_layers=1, dropout=0.0).to(DEVICE)

    history = train_model(model, train_loader, epochs=50, learning_rate=1e-3)

    y_pred, y_prob, metrics = evaluate_model(model, X_test_tensor, y_test_true)

    print("LSTM Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    save_metrics(metrics)
    update_comparison_table(metrics)
    save_model(model, scaler)
    plot_training_curve(history)
    plot_confusion(y_test_true, y_pred)
    plot_roc(y_test_true, y_prob)

    print("\nSaved:")
    print("- metrics to outputs/metrics/lstm_metrics.json")
    print("- model to outputs/models/lstm_model.pt")
    print("- scaler to outputs/models/lstm_scaler.pkl")
    print("- figures to outputs/figures/")


if __name__ == "__main__":
    main()