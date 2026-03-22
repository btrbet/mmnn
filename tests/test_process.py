# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
import csv
from pathlib import Path

import pytest

from mmnn.data.process import OUTPUT_COLUMNS, process_year


def test_process_year_creates_output_file(tmp_path: Path) -> None:
    """process_year creates {year}-data.csv in the data directory."""
    # Use bundled sample data by resolving from package
    data_dir = Path(__file__).resolve().parent.parent / "data" / "men"
    if not (data_dir / "2025-teams.csv").exists():
        pytest.skip("Sample data not found; run from project root")
    if not (data_dir / "2025-games.csv").exists():
        pytest.skip("Sample data not found; run from project root")

    process_year(2025, data_dir=data_dir)

    output_path = data_dir / "2025-data.csv"
    assert output_path.exists()


def test_process_year_output_has_expected_header(tmp_path: Path) -> None:
    """Output CSV has the expected column headers."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "men"
    if not (data_dir / "2025-teams.csv").exists():
        pytest.skip("Sample data not found; run from project root")

    process_year(2025, data_dir=data_dir)

    with (data_dir / "2025-data.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == OUTPUT_COLUMNS


def test_process_year_output_row_count_matches_games(tmp_path: Path) -> None:
    """Output row count equals number of games (all teams resolvable)."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "men"
    if not (data_dir / "2025-games.csv").exists():
        pytest.skip("Sample data not found; run from project root")

    with (data_dir / "2025-games.csv").open(newline="", encoding="utf-8") as f:
        game_count = sum(1 for _ in csv.DictReader(f))

    process_year(2025, data_dir=data_dir)

    with (data_dir / "2025-data.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == game_count


def test_process_year_first_game_values() -> None:
    '''
    First game: Mount St. Mary's vs American.
    Both Rank 16; tiebreak SRS: American -7.63 > Mount St. Mary's -7.91.
    So American is higher-ranked. Mount St. Mary's won -> Winner=LOWER.
    '''
    data_dir = Path(__file__).resolve().parent.parent / "data" / "men"
    if not (data_dir / "2025-teams.csv").exists():
        pytest.skip("Sample data not found; run from project root")

    process_year(2025, data_dir=data_dir)

    with (data_dir / "2025-data.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    first = rows[0]
    assert first["Winner"] == "LOWER"
    assert float(first["Δ Rank"]) == 0  # both rank 16
    assert float(first["Δ SRS"]) == pytest.approx(0.28, abs=0.01)  # -7.63 - (-7.91)
