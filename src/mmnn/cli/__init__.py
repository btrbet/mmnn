# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
import click


@click.group()
def mmnn():
    """NCAA Division I college basketball tournament bracket prediction (men's or women's)."""
    pass


@mmnn.group()
def data():
    """Data management commands."""
    pass


WOMEN_HELP = "Use NCAA Women's tournament data (data/women; Sports Reference women's URLs)."


@data.command()
@click.argument("year", type=int)
@click.option("-w", "--women", is_flag=True, help=WOMEN_HELP)
def fetch(year: int, women: bool) -> None:
    """Fetch NCAA tournament data for the specified year from Sports Reference."""
    from mmnn.data.fetch import fetch_year

    fetch_year(year, women=women)


@data.command()
@click.argument("year", type=int)
@click.option("-w", "--women", is_flag=True, help=WOMEN_HELP)
def process(year: int, women: bool) -> None:
    """Process raw data for the specified year."""
    from mmnn.data.process import process_year

    process_year(year, women=women)


@mmnn.group()
def nn():
    """Neural network training and prediction."""
    pass


@nn.command()
@click.option("-w", "--women", is_flag=True, help=WOMEN_HELP)
def train(women: bool) -> None:
    """Train the neural network and save weights to data/men/model.pt or data/women/model.pt."""
    from mmnn.nn.train import run_train

    run_train(women=women)


@nn.command()
@click.argument("year", type=int)
@click.option("-w", "--women", is_flag=True, help=WOMEN_HELP)
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="Training epochs when fitting on other years (default: same as mmnn nn train).",
)
def bracket(year: int, women: bool, epochs: int | None) -> None:
    """Predict each game in the year's bracket; print per-game results and evaluation metrics.

    Retrains on all *-data.csv rows except this year, then evaluates on {year}-games.csv
    (holdout for that tournament).
    """
    from mmnn.nn.bracket import run_bracket

    run_bracket(year, women=women, epochs=epochs)


@nn.command()
@click.argument("team1", type=str)
@click.argument("team2", type=str)
@click.option("-w", "--women", is_flag=True, help=WOMEN_HELP)
def predict(team1: str, team2: str, women: bool) -> None:
    """Predict which team wins. Team stats are looked up from data/men|women/2026-teams.csv."""
    from mmnn.nn.predict import run_predict

    run_predict(team1, team2, women=women)
