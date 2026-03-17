# mmnn

[![PyPI - Version](https://img.shields.io/pypi/v/mmnn.svg)](https://pypi.org/project/mmnn)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmnn.svg)](https://pypi.org/project/mmnn)

-----

This project creates a NCAA DIV I Men's College Basketball Tournament bracket prediction using a neural network.

This project was inspired by [this paper](https://github.com/btrbet/mmnn/blob/main/research/Comparing%20Various%20Machine%20Learning%20Statistical%20Methods%20Using%20Vari.pdf).

Tournament data for 2010 - 2026 is included in the repository. If you'd like to add additional years of tournament data to the training dataset, use the `mmnn data fetch` command.

## Usage

**Basic workflow:**
```bash
mmnn nn train
mmnn nn predict Duke Siena
```

### Fetch data

```
mmnn data fetch <year>
```
Fetch the raw data for the given year.

### Process data

Process raw data for a given year into the format needed by the neural network. Reads `data/YEAR-teams.csv` and `data/YEAR-games.csv`, then writes `data/YEAR-data.csv` with per-game delta features and a Winner label.

```bash
mmnn data process <year>
```

**Development (with Hatch):**
```bash
hatch run mmnn data process 2025
```

### Train the neural network

Train the model on all `data/*-data.csv` files (90% train / 10% test split), then save weights to `data/model.pt`:

```bash
mmnn nn train
```

### Use the neural network

Predict which team wins (higher- or lower-ranked) given two team names. Team stats are looked up from `data/2026-teams.csv`:

```bash
mmnn nn predict <team1> <team2>
mmnn nn predict "Ohio State" TCU
```
Look in `2026-teams.csv` for the correct team names to use.

## Installation

**From PyPI (end users):**

```console
pip install mmnn
```

**From source (development):**

This project uses [Hatch](https://hatch.pypa.io/) for dependency management and building. Clone the repo, then run commands without installing:

```console
hatch run mmnn data fetch 2024
hatch run mmnn data process 2024
hatch run mmnn nn train
hatch run mmnn nn predict Duke UConn
```

Or install in editable mode: `pip install -e .`

## License

`mmnn` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
