# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
import click


@click.group()
def mmnn():
    """NCAA Division I Men's College Basketball Tournament bracket prediction."""
    pass


@mmnn.group()
def data():
    """Data management commands."""
    pass


@data.command()
@click.argument("year", type=int)
def fetch(year: int) -> None:
    """Fetch NCAA tournament data for the specified year from Sports Reference."""
    from mmnn.data.fetch import fetch_year

    fetch_year(year)


@data.command()
@click.argument("year", type=int)
def process(year: int) -> None:
    """Process raw data for the specified year."""
    from mmnn.data.process import process_year

    process_year(year)


@mmnn.group()
def nn():
    """Neural network training and prediction."""
    pass


@nn.command()
def train() -> None:
    """Train the neural network and save weights to data/model.pt."""
    from mmnn.nn.train import run_train

    run_train()


@nn.command()
@click.argument("team1", type=str)
@click.argument("team2", type=str)
def predict(team1: str, team2: str) -> None:
    """Predict which team wins. Team stats are looked up from data/2026-teams.csv."""
    from mmnn.nn.predict import run_predict

    run_predict(team1, team2)
