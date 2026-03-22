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
@click.argument("team1", type=str)
@click.argument("team2", type=str)
@click.option("-w", "--women", is_flag=True, help=WOMEN_HELP)
def predict(team1: str, team2: str, women: bool) -> None:
    """Predict which team wins. Team stats are looked up from data/men|women/2026-teams.csv."""
    from mmnn.nn.predict import run_predict

    run_predict(team1, team2, women=women)
