# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
"""Project paths for tournament data (men vs women subdirectories)."""
from pathlib import Path


def project_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent


def data_dir(*, women: bool = False) -> Path:
    """Directory for CSVs and model weights: ``data/men`` or ``data/women``."""
    return project_root() / "data" / ("women" if women else "men")
