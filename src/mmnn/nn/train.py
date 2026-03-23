# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
import random
from pathlib import Path

import torch
from sklearn.preprocessing import StandardScaler

from mmnn import paths
from mmnn.nn.data import FEATURE_COLS, load_all_data_rows, rows_to_tensors
from mmnn.nn.metrics import binary_prediction_metrics
from mmnn.nn.model import BracketPredictor

MODEL_FILENAME = "model.pt"
DEFAULT_EPOCHS = 150
SEED = 42


def _get_model_path(data_dir: Path) -> Path:
    return data_dir / MODEL_FILENAME


def fit_model_on_rows(
    rows: list[dict],
    *,
    epochs: int = DEFAULT_EPOCHS,
    train_frac: float = 0.9,
    print_test_metrics: bool = True,
) -> tuple[BracketPredictor, StandardScaler]:
    """
    Train ``BracketPredictor`` on processed data rows.

    If train_frac < 1, hold out a random fraction for printed test metrics (when possible).
    If train_frac >= 1, train on all rows and skip the internal test split.
    """
    if not rows:
        raise ValueError("rows must be non-empty")

    random.seed(SEED)
    random.shuffle(rows)

    test_rows: list[dict] = []
    if train_frac >= 1.0:
        train_rows = rows
    elif len(rows) < 2:
        train_rows = rows
    else:
        split_idx = int(len(rows) * train_frac)
        split_idx = max(1, min(split_idx, len(rows) - 1))
        train_rows = rows[:split_idx]
        test_rows = rows[split_idx:]

    train_features, train_labels = rows_to_tensors(train_rows)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)

    if test_rows:
        test_features, test_labels = rows_to_tensors(test_rows)
        X_test = scaler.transform(test_features)
    else:
        test_features = None
        test_labels = None
        X_test = None

    model = BracketPredictor()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred.unsqueeze(1), y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    if (
        print_test_metrics
        and test_rows
        and test_features is not None
        and test_labels is not None
        and X_test is not None
    ):
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            preds = model(X_test_t)
            y_true = torch.tensor(test_labels, dtype=torch.float32)
            binary_prediction_metrics(preds, y_true, print_output=True)

    return model, scaler


def run_train(
    data_dir: Path | None = None,
    epochs: int = DEFAULT_EPOCHS,
    train_frac: float = 0.9,
    *,
    women: bool = False,
) -> None:
    """
    Load all YEAR-data.csv rows, shuffle, split 90/10, train model, save weights and scaler.
    """
    if data_dir is None:
        data_dir = paths.data_dir(women=women)

    rows = load_all_data_rows(data_dir)
    if not rows:
        raise SystemExit("No *-data.csv files found or no valid rows. Run 'mmnn data process <year>' first.")

    model, scaler = fit_model_on_rows(
        rows,
        epochs=epochs,
        train_frac=train_frac,
        print_test_metrics=True,
    )

    model_path = _get_model_path(data_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "feature_order": FEATURE_COLS,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")
