# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
import csv
from pathlib import Path

from mmnn.data.process import OUTPUT_COLUMNS, compute_deltas_for_two_teams

FEATURE_COLS = [c for c in OUTPUT_COLUMNS if c != "Winner"]


def _get_data_dir() -> Path:
    """Resolve data directory relative to project root."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    return project_root / "data"


def load_all_data_rows(data_dir: Path | None = None) -> list[dict]:
    """
    Glob data/*-data.csv, load all rows, return list of dicts.
    Filters out rows with invalid numeric values.
    """
    if data_dir is None:
        data_dir = _get_data_dir()
    rows: list[dict] = []
    for path in sorted(data_dir.glob("*-data.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    for col in FEATURE_COLS:
                        float(row[col])
                    if row["Winner"] not in ("HIGHER", "LOWER"):
                        continue
                    rows.append(row)
                except (ValueError, KeyError):
                    continue
    return rows


def rows_to_tensors(
    rows: list[dict],
) -> tuple[list[list[float]], list[int]]:
    """
    Extract feature vectors and labels from data rows.
    Returns (features_list, labels_list) where label 1 = HIGHER wins, 0 = LOWER wins.
    """
    features = []
    labels = []
    for row in rows:
        feat = [float(row[c]) for c in FEATURE_COLS]
        features.append(feat)
        labels.append(1 if row["Winner"] == "HIGHER" else 0)
    return features, labels


def _load_teams_lookup(path: Path) -> dict[str, dict]:
    """Load teams CSV and build case-insensitive Team -> row lookup."""
    lookup: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["Team"].strip().lower()
            if key not in lookup:
                lookup[key] = row
    return lookup


def _resolve_team_by_name(lookup: dict[str, dict], name: str) -> dict | None:
    """Resolve team name (case-insensitive) to team row, or None if not found."""
    return lookup.get(name.strip().lower())


def compute_deltas_from_team_names(
    team1: str, team2: str, teams_path: Path | None = None
) -> tuple[list[float], dict, dict]:
    """
    Look up two teams by name in teams CSV, compute delta features.
    Returns (deltas, higher_row, lower_row) where deltas is the 9-dimensional
    feature vector (higher - lower).
    Uses data/2026-teams.csv by default.
    """
    if teams_path is None:
        teams_path = _get_data_dir() / "2026-teams.csv"
    teams_path = Path(teams_path)
    if not teams_path.exists():
        raise FileNotFoundError(f"Teams file not found: {teams_path}")

    lookup = _load_teams_lookup(teams_path)
    row1 = _resolve_team_by_name(lookup, team1)
    row2 = _resolve_team_by_name(lookup, team2)

    if row1 is None:
        raise ValueError(f"Team '{team1}' not found in {teams_path}")
    if row2 is None:
        raise ValueError(f"Team '{team2}' not found in {teams_path}")

    deltas, higher_row, lower_row = compute_deltas_for_two_teams(row1, row2)
    return deltas, higher_row, lower_row


def compute_deltas_from_teams_file(path: str | Path) -> list[float]:
    """
    Read a CSV with exactly 2 team rows (teams.csv format), compute delta features.
    Returns the 9-dimensional feature vector (higher - lower).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 2:
        raise ValueError(f"Expected exactly 2 team rows, got {len(rows)}")
    deltas, _, _ = compute_deltas_for_two_teams(rows[0], rows[1])
    return deltas
