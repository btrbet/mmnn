# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
"""Evaluation metrics for binary HIGHER/LOWER predictions."""
from __future__ import annotations

import math
from typing import Any

import torch


def binary_prediction_metrics(
    preds: torch.Tensor,
    y_true: torch.Tensor,
    *,
    print_output: bool = True,
) -> dict[str, Any]:
    """
    Compute accuracy, R², RMSE, MAD, NLL, etc. for probabilities of class 1 (HIGHER wins).

    preds: predicted P(HIGHER), shape (n,) or (n, 1).
    y_true: 0/1 labels, same length as preds.
    """
    preds = preds.flatten()
    y_true = y_true.flatten().float()
    n = int(preds.shape[0])
    if n == 0:
        out: dict[str, Any] = {
            "n": 0,
            "accuracy": 0.0,
            "correct": 0,
            "misclassification_rate": 0.0,
            "rmse": 0.0,
            "mad": 0.0,
            "nll": 0.0,
            "generalized_r2": 0.0,
            "entropy_r2": 0.0,
            "sum_freq": 0,
        }
        if print_output:
            print("No samples to evaluate.")
        return out

    preds_clamped = preds.clamp(min=1e-7, max=1 - 1e-7)

    correct = int(((preds >= 0.5).int() == y_true.int()).sum().item())
    acc = correct / n
    misclass_rate = 1 - acc

    rmse = ((preds - y_true).pow(2).mean().sqrt()).item()
    mad = (preds - y_true).abs().mean().item()

    nll = -(y_true * preds_clamped.log() + (1 - y_true) * (1 - preds_clamped).log()).sum().item()

    p_null = y_true.mean().item()
    p_null = max(1e-7, min(1 - 1e-7, p_null))
    nll_null_val = -n * (p_null * math.log(p_null) + (1 - p_null) * math.log(1 - p_null))

    gen_r2 = 1 - (nll / nll_null_val) if nll_null_val != 0 else 0.0
    entropy_r2 = 1 - (2 * nll) / (2 * nll_null_val) if nll_null_val != 0 else 0.0

    out = {
        "n": n,
        "accuracy": acc,
        "correct": correct,
        "misclassification_rate": misclass_rate,
        "rmse": rmse,
        "mad": mad,
        "nll": nll,
        "generalized_r2": gen_r2,
        "entropy_r2": entropy_r2,
        "sum_freq": n,
    }

    if print_output:
        print(f"Test accuracy: {acc:.2%} ({correct}/{n})")
        print(f"Generalized R²: {gen_r2:.4f}")
        print(f"Entropy R²: {entropy_r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Mean Abs Dev: {mad:.4f}")
        print(f"Misclassification rate: {misclass_rate:.2%}")
        print(f"-LogLikelihood: {nll:.4f}")
        print(f"Sum freq: {n}")

    return out
