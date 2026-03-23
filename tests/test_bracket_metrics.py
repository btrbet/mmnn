# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
"""Tests for mmnn.nn.metrics and bracket CLI."""
import math

import pytest
import torch
from click.testing import CliRunner

from mmnn import paths
from mmnn.cli import mmnn
from mmnn.nn.data import load_all_data_rows
from mmnn.nn.metrics import binary_prediction_metrics


def test_binary_prediction_metrics_empty() -> None:
    """Empty tensors yield zero counts and no crash."""
    out = binary_prediction_metrics(torch.tensor([]), torch.tensor([]), print_output=False)
    assert out["n"] == 0
    assert out["correct"] == 0


def test_binary_prediction_metrics_perfect() -> None:
    """All correct when preds and labels align."""
    preds = torch.tensor([0.9, 0.1, 0.99])
    y_true = torch.tensor([1.0, 0.0, 1.0])
    out = binary_prediction_metrics(preds, y_true, print_output=False)
    assert out["n"] == 3
    assert out["correct"] == 3
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["misclassification_rate"] == pytest.approx(0.0)


def test_binary_prediction_metrics_nll_two_samples() -> None:
    """-LogLikelihood matches manual Bernoulli formula for two points."""
    preds = torch.tensor([0.5, 0.5])
    y_true = torch.tensor([1.0, 0.0])
    out = binary_prediction_metrics(preds, y_true, print_output=False)
    # p clamped to 0.5; each term -log(0.5); sum = 2*log(2)
    expected_nll = 2 * math.log(2)
    assert out["nll"] == pytest.approx(expected_nll, rel=1e-5)


def test_load_all_data_rows_exclude_year() -> None:
    """exclude_year omits that year's file from the combined row list."""
    data_dir = paths.data_dir(women=True)
    all_rows = load_all_data_rows(data_dir, women=True)
    if not all_rows:
        pytest.skip("no *-data.csv in women data dir")
    no_missing_year = load_all_data_rows(data_dir, women=True, exclude_year=3000)
    assert len(no_missing_year) == len(all_rows)
    if (data_dir / "2025-data.csv").exists():
        without_2025 = load_all_data_rows(data_dir, women=True, exclude_year=2025)
        assert len(without_2025) <= len(all_rows)


def test_mmnn_nn_bracket_cli() -> None:
    """End-to-end CLI retrains (holdout year) and evaluates bracket."""
    data_dir = paths.data_dir(women=True)
    if not (data_dir / "2025-games.csv").exists():
        pytest.skip("2025 games not present")
    if not load_all_data_rows(data_dir, women=True, exclude_year=2025):
        pytest.skip("need at least one other *-data.csv besides 2025 for holdout training")

    runner = CliRunner()
    result = runner.invoke(mmnn, ["nn", "bracket", "2025", "--women", "--epochs", "2"])
    assert result.exit_code == 0, result.output + getattr(result, "stderr", "")
    assert "2025 tournament" in result.output
    assert "holdout evaluation" in result.output
    assert "P(HIGHER)" in result.output
    assert "Test accuracy:" in result.output
