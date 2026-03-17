# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
import math
import random
from pathlib import Path

import torch
from sklearn.preprocessing import StandardScaler

from mmnn.nn.data import FEATURE_COLS, load_all_data_rows, rows_to_tensors
from mmnn.nn.model import BracketPredictor

MODEL_FILENAME = "model.pt"
DEFAULT_EPOCHS = 150
SEED = 42


def _get_data_dir() -> Path:
    """Resolve data directory relative to project root."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    return project_root / "data"


def _get_model_path(data_dir: Path) -> Path:
    return data_dir / MODEL_FILENAME


def run_train(
    data_dir: Path | None = None,
    epochs: int = DEFAULT_EPOCHS,
    train_frac: float = 0.9,
) -> None:
    """
    Load all YEAR-data.csv rows, shuffle, split 90/10, train model, save weights and scaler.
    """
    if data_dir is None:
        data_dir = _get_data_dir()

    rows = load_all_data_rows(data_dir)
    if not rows:
        raise SystemExit("No *-data.csv files found or no valid rows. Run 'mmnn data process <year>' first.")

    random.seed(SEED)
    random.shuffle(rows)

    split_idx = int(len(rows) * train_frac)
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]

    train_features, train_labels = rows_to_tensors(train_rows)
    test_features, test_labels = rows_to_tensors(test_rows)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)

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

    # Test-set evaluation
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_test_t)
        y_true = torch.tensor(test_labels, dtype=torch.float32)
        n = len(test_labels)

        # Clamp preds for log stability
        preds_clamped = preds.clamp(min=1e-7, max=1 - 1e-7)

        # Accuracy / misclassification
        correct = ((preds >= 0.5).int() == y_true.int()).sum().item()
        acc = correct / n if n else 0
        misclass_rate = 1 - acc

        # RMSE
        rmse = ((preds - y_true).pow(2).mean().sqrt()).item()

        # Mean Absolute Deviation
        mad = (preds - y_true).abs().mean().item()

        # -LogLikelihood (negative binomial log-likelihood for binary)
        nll = -(y_true * preds_clamped.log() + (1 - y_true) * (1 - preds_clamped).log()).sum().item()

        # Null model log-likelihood (predict mean for all)
        p_null = y_true.mean().item()
        p_null = max(1e-7, min(1 - 1e-7, p_null))
        nll_null_val = -n * (p_null * math.log(p_null) + (1 - p_null) * math.log(1 - p_null))

        # Generalized R² (McFadden): 1 - (LL_model / LL_null) = 1 - (nll / nll_null) but nll is -LL
        # LL_model = -nll, LL_null = -nll_null_val. So R2 = 1 - (-nll / -nll_null_val) = 1 - nll/nll_null_val
        gen_r2 = 1 - (nll / nll_null_val) if nll_null_val != 0 else 0.0

        # Entropy R²: 1 - (deviance / null_deviance), deviance = 2*nll
        entropy_r2 = 1 - (2 * nll) / (2 * nll_null_val) if nll_null_val != 0 else 0.0

        # Sum freq (total observations)
        sum_freq = n

        print(f"Test accuracy: {acc:.2%} ({correct}/{n})")
        print(f"Generalized R²: {gen_r2:.4f}")
        print(f"Entropy R²: {entropy_r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Mean Abs Dev: {mad:.4f}")
        print(f"Misclassification rate: {misclass_rate:.2%}")
        print(f"-LogLikelihood: {nll:.4f}")
        print(f"Sum freq: {sum_freq}")

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
