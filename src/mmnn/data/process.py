# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
import csv
import sys
from pathlib import Path

# Common team name typos in games CSV (games -> teams)
TEAM_NAME_ALIASES = {
    "Houson": "Houston",
}

STAT_COLUMNS = ["Rank", "SRS", "SOS", "WL%", "FG/G", "OREB/G", "TS%", "TOV%", "AST%"]
OUTPUT_COLUMNS = [
    "Winner",
    "Δ Rank",
    "Δ SRS",
    "Δ SOS",
    "Δ WL%",
    "Δ FG/G",
    "Δ OREB/G",
    "Δ TS%",
    "Δ TOV%",
    "Δ AST%",
]


def _get_data_dir() -> Path:
    """Resolve data directory relative to project root (parent of src/)."""
    # process.py lives at src/mmnn/data/process.py -> 4 levels up to project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    return project_root / "data"


def _load_teams(data_dir: Path, year: int) -> dict[str, dict]:
    """Load teams CSV and build Team -> row lookup. Uses first match for duplicates."""
    teams_path = data_dir / f"{year}-teams.csv"
    if not teams_path.exists():
        raise FileNotFoundError(f"Teams file not found: {teams_path}")

    lookup: dict[str, dict] = {}
    with teams_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = row["Team"].strip()
            if team not in lookup:
                lookup[team] = row
    return lookup


def _normalize_team_name(name: str) -> str:
    """Apply known aliases for team name typos."""
    return TEAM_NAME_ALIASES.get(name.strip(), name.strip())


def _resolve_team(teams: dict[str, dict], name: str) -> dict | None:
    """Resolve team name (with aliases) to team row, or None if not found."""
    normalized = _normalize_team_name(name)
    return teams.get(normalized)


def _is_higher_ranked(row1: dict, row2: dict) -> bool:
    """
    Return True if row1 is higher-ranked than row2.
    Tiebreaker: Rank (lower=better) -> SRS (higher=better) -> SOS (higher=better) -> True (row1)
    """
    r1, r2 = int(row1["Rank"]), int(row2["Rank"])
    if r1 != r2:
        return r1 < r2  # lower rank = better

    s1, s2 = float(row1["SRS"]), float(row2["SRS"])
    if s1 != s2:
        return s1 > s2  # higher SRS = better

    s1, s2 = float(row1["SOS"]), float(row2["SOS"])
    if s1 != s2:
        return s1 > s2  # higher SOS = better

    return True  # row1 wins tiebreak


def _compute_deltas(higher: dict, lower: dict) -> list[float]:
    """Compute (higher - lower) for each stat in STAT_COLUMNS."""
    result = []
    for col in STAT_COLUMNS:
        v1 = float(higher[col])
        v2 = float(lower[col])
        result.append(v1 - v2)
    return result


def compute_deltas_for_two_teams(row1: dict, row2: dict) -> tuple[list[float], dict, dict]:
    """
    Compute delta features for two team rows.
    Returns (deltas, higher_row, lower_row) where deltas = higher - lower for each stat.
    """
    if _is_higher_ranked(row1, row2):
        higher, lower = row1, row2
    else:
        higher, lower = row2, row1
    return _compute_deltas(higher, lower), higher, lower


def process_year(year: int, data_dir: Path | None = None) -> None:
    """
    Process raw data for the specified year.

    Reads data/{year}-teams.csv and data/{year}-games.csv, joins them to compute
    per-game deltas and Winner label, and writes data/{year}-data.csv.
    """
    if data_dir is None:
        data_dir = _get_data_dir()

    teams = _load_teams(data_dir, year)
    games_path = data_dir / f"{year}-games.csv"
    if not games_path.exists():
        raise FileNotFoundError(f"Games file not found: {games_path}")

    output_path = data_dir / f"{year}-data.csv"
    rows: list[dict[str, str | float]] = []

    with games_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, game in enumerate(reader):
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

            # Winner: "HIGHER" if higher-ranked team won, else "LOWER"
            higher_team_name = _normalize_team_name(higher["Team"])
            lower_team_name = _normalize_team_name(lower["Team"])
            if winner_name == higher_team_name:
                winner_label = "HIGHER"
            elif winner_name == lower_team_name:
                winner_label = "LOWER"
            else:
                print(
                    f"Warning: Winner '{game['Winner']}' does not match either team, skipping game",
                    file=sys.stderr,
                )
                continue

            deltas = _compute_deltas(higher, lower)
            rows.append(
                {
                    "Winner": winner_label,
                    "Δ Rank": deltas[0],
                    "Δ SRS": deltas[1],
                    "Δ SOS": deltas[2],
                    "Δ WL%": deltas[3],
                    "Δ FG/G": deltas[4],
                    "Δ OREB/G": deltas[5],
                    "Δ TS%": deltas[6],
                    "Δ TOV%": deltas[7],
                    "Δ AST%": deltas[8],
                }
            )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
