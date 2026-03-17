# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
"""Tests for mmnn neural network module."""
from pathlib import Path

import pytest

from mmnn.data.process import compute_deltas_for_two_teams
from mmnn.nn.data import (
    FEATURE_COLS,
    compute_deltas_from_team_names,
    compute_deltas_from_teams_file,
    load_all_data_rows,
    rows_to_tensors,
)
from mmnn.nn.model import BracketPredictor
from mmnn.nn.predict import run_predict
from mmnn.nn.train import run_train


def test_compute_deltas_from_teams_file_matches_process() -> None:
    """compute_deltas_from_teams_file produces same deltas as process.py for a known game."""
    fixture_path = (
        Path(__file__).resolve().parent / "fixtures" / "two-teams-uconn-stetson.csv"
    )
    if not fixture_path.exists():
        pytest.skip("Fixture not found")

    deltas = compute_deltas_from_teams_file(fixture_path)
    assert len(deltas) == 9
    # First game in 2024-data.csv: UConn (1) vs Stetson (16), HIGHER won
    # Δ Rank = 1 - 16 = -15, Δ SRS = 26.7 - (-5.09) ≈ 31.79
    assert deltas[0] == pytest.approx(-15.0, abs=0.01)
    assert deltas[1] == pytest.approx(31.79, abs=0.01)


def test_compute_deltas_from_teams_file_row_order_independent() -> None:
    """Delta computation yields same result regardless of row order (always higher - lower)."""
    fixture_path = (
        Path(__file__).resolve().parent / "fixtures" / "two-teams-uconn-stetson.csv"
    )
    if not fixture_path.exists():
        pytest.skip("Fixture not found")

    import csv

    with fixture_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    deltas_ab, _, _ = compute_deltas_for_two_teams(rows[0], rows[1])
    deltas_ba, _, _ = compute_deltas_for_two_teams(rows[1], rows[0])
    # compute_deltas_for_two_teams always returns (higher - lower); order of args is irrelevant
    for i, (a, b) in enumerate(zip(deltas_ab, deltas_ba)):
        assert a == pytest.approx(b, abs=1e-9), f"Column {i} should match"


def test_compute_deltas_from_team_names() -> None:
    """compute_deltas_from_team_names looks up teams in 2026-teams.csv and returns 9 deltas."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    teams_path = data_dir / "2026-teams.csv"
    if not teams_path.exists():
        pytest.skip("2026-teams.csv not found")

    deltas, higher_row, lower_row = compute_deltas_from_team_names(
        "Duke", "UConn", teams_path
    )
    assert len(deltas) == 9
    assert "Team" in higher_row and "Team" in lower_row


def test_compute_deltas_from_team_names_case_insensitive() -> None:
    """Team name lookup is case-insensitive."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    teams_path = data_dir / "2026-teams.csv"
    if not teams_path.exists():
        pytest.skip("2026-teams.csv not found")

    deltas_upper, _, _ = compute_deltas_from_team_names("DUKE", "UCONN", teams_path)
    deltas_lower, _, _ = compute_deltas_from_team_names("duke", "uconn", teams_path)
    assert deltas_upper == deltas_lower


def test_compute_deltas_from_team_names_raises_when_team_not_found() -> None:
    """compute_deltas_from_team_names raises if either team is not in the file."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    teams_path = data_dir / "2026-teams.csv"
    if not teams_path.exists():
        pytest.skip("2026-teams.csv not found")

    with pytest.raises(ValueError, match="Team 'NonexistentTeam' not found"):
        compute_deltas_from_team_names("Duke", "NonexistentTeam", teams_path)


def test_load_all_data_rows() -> None:
    """load_all_data_rows returns rows from *-data.csv files."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    rows = load_all_data_rows(data_dir)
    if not rows:
        pytest.skip("No *-data.csv files found")
    for row in rows:
        assert "Winner" in row
        assert row["Winner"] in ("HIGHER", "LOWER")
        for col in FEATURE_COLS:
            assert col in row
            float(row[col])


def test_rows_to_tensors() -> None:
    """rows_to_tensors extracts features and labels correctly."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    rows = load_all_data_rows(data_dir)
    if not rows:
        pytest.skip("No *-data.csv files found")
    features, labels = rows_to_tensors(rows[:10])
    assert len(features) == len(labels)
    assert len(features[0]) == 9
    assert all(l in (0, 1) for l in labels)


def test_bracket_predictor_forward() -> None:
    """BracketPredictor forward returns probabilities in [0, 1]."""
    import torch

    model = BracketPredictor()
    x = torch.randn(4, 9)
    out = model(x)
    assert out.shape == (4,)
    assert (out >= 0).all() and (out <= 1).all()


def test_nn_train_creates_model_file() -> None:
    """mmnn nn train runs and creates data/model.pt."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    rows = load_all_data_rows(data_dir)
    if not rows:
        pytest.skip("No *-data.csv files found")

    run_train(data_dir=data_dir, epochs=2)

    model_path = data_dir / "model.pt"
    assert model_path.exists()


def test_nn_predict_outputs_high_or_low() -> None:
    """mmnn nn predict with two team names outputs Result line and metrics."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    model_path = data_dir / "model.pt"
    teams_path = data_dir / "2026-teams.csv"
    if not model_path.exists() or not teams_path.exists():
        pytest.skip("Model or 2026-teams.csv not found")

    import io
    import sys

    cap = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = cap
    try:
        run_predict("Duke", "UConn", data_dir=data_dir)
        output = cap.getvalue().strip()
    finally:
        sys.stdout = old_stdout

    assert "Result:" in output
    assert "HIGHER" in output or "LOWER" in output
    assert "Confidence:" in output
