# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
from pathlib import Path

from mmnn.paths import data_dir, project_root


def test_project_root_contains_src() -> None:
    assert (project_root() / "src" / "mmnn").is_dir()


def test_data_dir_men_and_women() -> None:
    root = project_root()
    assert data_dir(women=False) == root / "data" / "men"
    assert data_dir(women=True) == root / "data" / "women"
