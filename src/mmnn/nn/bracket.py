# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
"""Run NN on a year's bracket after training on all other years' processed data."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

from mmnn import paths
from mmnn.data.process import (
    _is_higher_ranked,
    _load_teams,
    _normalize_team_name,
    _resolve_team,
    compute_deltas_for_two_teams,
)
from mmnn.nn.data import FEATURE_COLS, load_all_data_rows
from mmnn.nn.metrics import binary_prediction_metrics
from mmnn.nn.model import BracketPredictor
from mmnn.nn.train import DEFAULT_EPOCHS, fit_model_on_rows


def _deltas_to_feature_vector(
    deltas: list[float],
    feature_order: list[str],
) -> list[float]:
    """Map deltas (FEATURE_COLS order) to feature_order for the saved scaler."""
    delta_by_name = dict(zip(FEATURE_COLS, deltas))
    return [delta_by_name[f] for f in feature_order]


def run_bracket(
    year: int,
    data_dir: Path | None = None,
    *,
    women: bool = False,
    epochs: int | None = None,
) -> None:
    """
    Retrain on every *-data.csv except this year, then predict each game in {year}-games.csv.

    Evaluation is out-of-sample for that tournament (training rows exclude ``{year}-data.csv``).
    """
    if data_dir is None:
        data_dir = paths.data_dir(women=women)

    if epochs is None:
        epochs = DEFAULT_EPOCHS

    train_rows = load_all_data_rows(data_dir, women=women, exclude_year=year)
    if not train_rows:
        raise SystemExit(
            f"No training rows after excluding {year}-data.csv. "
            "Process at least one other year with 'mmnn data process <year>' first."
        )

    games_path = data_dir / f"{year}-games.csv"
    if not games_path.exists():
        raise SystemExit(f"Games file not found: {games_path}")

    teams = _load_teams(data_dir, year)

    rows_meta: list[dict[str, str | int]] = []
    feature_rows: list[list[float]] = []
    labels: list[int] = []

    with games_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for game in reader:
            t1_name = game["Team 1"].strip()
            t2_name = game["Team 2"].strip()
            winner_name = _normalize_team_name(game["Winner"])

            row1 = _resolve_team(teams, t1_name)
            row2 = _resolve_team(teams, t2_name)

            if row1 is None:
                print(
                    f"Warning: Team 1 '{t1_name}' not found in teams CSV, skipping game",
                    file=sys.stderr,
                )
                continue
            if row2 is None:
                print(
                    f"Warning: Team 2 '{t2_name}' not found in teams CSV, skipping game",
                    file=sys.stderr,
                )
                continue

            if _is_higher_ranked(row1, row2):
                higher, lower = row1, row2
            else:
                higher, lower = row2, row1

            higher_team_name = _normalize_team_name(higher["Team"])
            lower_team_name = _normalize_team_name(lower["Team"])
            if winner_name == higher_team_name:
                y = 1
            elif winner_name == lower_team_name:
                y = 0
            else:
                print(
                    f"Warning: Winner '{game['Winner']}' does not match either team, skipping game",
                    file=sys.stderr,
                )
                continue

            deltas, higher_row, lower_row = compute_deltas_for_two_teams(row1, row2)

            rows_meta.append(
                {
                    "team1": t1_name,
                    "team2": t2_name,
                    "winner": winner_name,
                    "higher_name": higher_row["Team"],
                    "lower_name": lower_row["Team"],
                    "y": y,
                }
            )
            feature_rows.append(deltas)
            labels.append(y)

    if not feature_rows:
        raise SystemExit("No games to evaluate (all skipped or empty bracket).")

    print(
        f"Training on {len(train_rows)} games from other years "
        f"(excluding {year}-data.csv), {epochs} epochs...",
        file=sys.stderr,
    )
    model, scaler = fit_model_on_rows(
        train_rows,
        epochs=epochs,
        train_frac=1.0,
        print_test_metrics=False,
    )

    feature_order = list(FEATURE_COLS)
    mean = scaler.mean_.tolist()
    scale = scaler.scale_.tolist()

    normalized: list[list[float]] = []
    for deltas in feature_rows:
        vec = _deltas_to_feature_vector(deltas, feature_order)
        normalized.append([(v - m) / s for v, m, s in zip(vec, mean, scale)])

    x = torch.tensor(normalized, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        preds = model(x)

    y_true = torch.tensor(labels, dtype=torch.float32)

    print(f"\n{year} tournament — {len(rows_meta)} games (holdout evaluation)\n")

    for i, meta in enumerate(rows_meta):
        ph = float(preds[i].item())
        pred_side = "HIGHER" if ph >= 0.5 else "LOWER"
        pred_team = str(meta["higher_name"] if pred_side == "HIGHER" else meta["lower_name"])
        y_i = int(meta["y"])
        correct = (1 if ph >= 0.5 else 0) == y_i
        mark = "ok" if correct else "miss"
        print(
            f"{meta['team1']} vs {meta['team2']} | "
            f"actual: {meta['winner']} | "
            f"pred: {pred_team} ({pred_side}, P(HIGHER)={ph:.2%}) | {mark}"
        )

    print()
    binary_prediction_metrics(preds, y_true, print_output=True)
