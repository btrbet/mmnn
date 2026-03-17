# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
from pathlib import Path

import torch

from mmnn.nn.data import FEATURE_COLS
from mmnn.nn.model import BracketPredictor

MODEL_FILENAME = "model.pt"


def _get_data_dir() -> Path:
    """Resolve data directory relative to project root."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    return project_root / "data"


def run_predict(
    team1: str, team2: str, data_dir: Path | None = None, teams_path: Path | None = None
) -> None:
    """
    Load model, look up two teams by name in data/2026-teams.csv, compute deltas, predict HIGHER or LOWER.
    """
    from mmnn.nn.data import compute_deltas_from_team_names

    if data_dir is None:
        data_dir = _get_data_dir()
    model_path = data_dir / MODEL_FILENAME
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}. Run 'mmnn nn train' first.")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    mean = checkpoint["scaler_mean"]
    scale = checkpoint["scaler_scale"]
    feature_order = checkpoint["feature_order"]

    if teams_path is None:
        teams_path = data_dir / "2026-teams.csv"
    deltas, higher_row, lower_row = compute_deltas_from_team_names(
        team1, team2, teams_path
    )
    # Deltas are in FEATURE_COLS order; reorder if saved order differs (future-proofing)
    if list(feature_order) != list(FEATURE_COLS):
        vec = [deltas[feature_order.index(c)] for c in FEATURE_COLS]
    else:
        vec = deltas

    normalized = [(v - m) / s for v, m, s in zip(vec, mean, scale)]
    x = torch.tensor([normalized], dtype=torch.float32)

    model = BracketPredictor()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        prob = model(x).item()

    result = "HIGHER" if prob >= 0.5 else "LOWER"
    winning_team = higher_row["Team"] if result == "HIGHER" else lower_row["Team"]
    confidence = prob if result == "HIGHER" else (1 - prob)

    print(f"Result: {winning_team} - {result}")
    print(f"Confidence: {confidence:.0%}")
