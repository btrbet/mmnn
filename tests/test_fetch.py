# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
"""Tests for mmnn data fetch."""
import csv
from pathlib import Path

import pytest

from mmnn.data.fetch import (
    BracketGame,
    BracketTeam,
    GAMES_CSV_COLUMNS,
    TEAMS_CSV_COLUMNS,
    _parse_bracket,
)


def test_parse_bracket_extracts_games_and_teams() -> None:
    """_parse_bracket extracts games and team seeds from HTML fixture."""
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    html_path = fixture_dir / "2023-bracket-snippet.html"
    if not html_path.exists():
        pytest.skip("Fixture not found")
    html = html_path.read_text(encoding="utf-8")

    games, team_seeds, team_school_ids = _parse_bracket(html, 2023)

    # Should have multiple games (Purdue-FDU, Memphis-FAU, FDU-Texas Southern First Four)
    assert len(games) >= 2
    # First game: 1 Purdue 58 vs 16 FDU 63
    g0 = games[0]
    assert isinstance(g0, BracketGame)
    assert g0.team1.name == "Purdue"
    assert g0.team1.seed == 1
    assert g0.team1.score == 58
    assert g0.team2.name == "FDU"
    assert g0.team2.seed == 16
    assert g0.team2.score == 63
    # Team seeds
    assert team_seeds["Purdue"] == 1
    assert team_seeds["FDU"] == 16
    assert team_seeds["Memphis"] == 8
    assert team_seeds["Florida Atlantic"] == 9
    # School IDs for URL
    assert team_school_ids["Purdue"] == "purdue"
    assert team_school_ids["FDU"] == "fairleigh-dickinson"
    assert team_school_ids["Texas Southern"] == "texas-southern"


def test_parse_bracket_games_have_scores() -> None:
    """Each game has both teams with scores (winner determinable)."""
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    html_path = fixture_dir / "2023-bracket-snippet.html"
    if not html_path.exists():
        pytest.skip("Fixture not found")
    html = html_path.read_text(encoding="utf-8")

    games, _, _ = _parse_bracket(html, 2023)

    for g in games:
        assert g.team1.score is not None
        assert g.team2.score is not None
        winner = g.team1.name if g.team1.score > g.team2.score else g.team2.name
        assert winner in (g.team1.name, g.team2.name)


def test_teams_csv_columns_match_schema() -> None:
    """TEAMS_CSV_COLUMNS matches expected schema from process.py."""
    expected = [
        "ID", "Year", "Team", "Rank", "Wins", "Losses", "WL%",
        "SOS", "SRS", "FG/G", "OREB/G", "Total Points", "FGA", "FTA",
        "AST", "TOV", "TS%", "TOV%", "AST%", "URL",
    ]
    assert TEAMS_CSV_COLUMNS == expected


def test_games_csv_columns_match_schema() -> None:
    """GAMES_CSV_COLUMNS matches expected schema for process.py."""
    expected = ["Team 1", "Team 1 Score", "Team 2", "Team 2 Score", "Winner"]
    assert GAMES_CSV_COLUMNS == expected


def test_bracket_team_dataclass() -> None:
    """BracketTeam stores name, seed, school_id, and optional score."""
    bt = BracketTeam(name="Duke", seed=1, school_id="duke", score=74)
    assert bt.name == "Duke"
    assert bt.seed == 1
    assert bt.school_id == "duke"
    assert bt.score == 74
