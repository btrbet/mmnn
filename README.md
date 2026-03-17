# mmnn

[![PyPI - Version](https://img.shields.io/pypi/v/mmnn.svg)](https://pypi.org/project/mmnn)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmnn.svg)](https://pypi.org/project/mmnn)

-----

This project creates a NCAA DIV I Men's College Basketball Tournament bracket prediction using a neural network.

## Usage

### Process data

Process raw data for a given year into the format needed by the neural network. Reads `data/YEAR-teams.csv` and `data/YEAR-games.csv`, then writes `data/YEAR-data.csv` with per-game delta features and a Winner label.

```bash
mmnn data process <year>
```

**Development (with Hatch):**
```bash
hatch run mmnn data process 2025
```

### Neural network

Train the model on all `data/*-data.csv` files (90% train / 10% test split), then save weights to `data/model.pt`:

```bash
mmnn nn train
```

Predict which team wins (higher- or lower-ranked) given two team names. Team stats are looked up from `data/2026-teams.csv`:

```bash
mmnn nn predict <team1> <team2>
```

**Example workflow:**
```bash
mmnn data process 2024
mmnn nn train
mmnn nn predict Duke UConn
```

### Fetch data

```
mmnn data fetch <year>
```
Fetch the raw data for the given year.

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install mmnn
```

## License

`mmnn` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
