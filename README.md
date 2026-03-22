# mmnn

[![PyPI - Version](https://img.shields.io/pypi/v/mmnn.svg)](https://pypi.org/project/mmnn)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmnn.svg)](https://pypi.org/project/mmnn)

-----

This project creates NCAA Division I college basketball tournament bracket predictions using a neural network (men's or women's).

This project was inspired by [this paper](https://github.com/btrbet/mmnn/blob/main/research/Comparing%20Various%20Machine%20Learning%20Statistical%20Methods%20Using%20Vari.pdf).

Tournament data for 2010–2026 (men's) is included under `data/men/`. To add more years, use `mmnn data fetch`. Use `-w` / `--women` on any command to use women's data under `data/women/` (Sports Reference URLs use `women` instead of `men` in the path).

## Usage

**Basic workflow (men's, default):**
```bash
mmnn nn train
mmnn nn predict Duke Siena
```

**Women's tournament** — add `-w` or `--women` to any command:

```bash
mmnn data fetch 2026 --women
mmnn data process 2026 --women
mmnn nn train --women
mmnn nn predict "South Carolina" UConn --women
```

### Fetch data

```
mmnn data fetch <year> [--women]
```

Fetch the raw bracket and team stats for the given year from [Sports Reference](https://www.sports-reference.com/cbb/).

### Process data

Process raw data for a given year into the format needed by the neural network. Reads `data/men|women/YEAR-teams.csv` and `YEAR-games.csv`, then writes `YEAR-data.csv` with per-game delta features and a Winner label.

```bash
mmnn data process <year>
```

**Development (with Hatch):**
```bash
hatch run mmnn data process 2025
```

### Train the neural network

Train the model on all `*-data.csv` files in `data/men/` or `data/women/` (90% train / 10% test split), then save weights to `data/men/model.pt` or `data/women/model.pt`:

```bash
mmnn nn train
mmnn nn train --women
```

### Use the neural network

Predict which team wins (higher- or lower-ranked) given two team names. Team stats are looked up from `data/men/2026-teams.csv` (or `data/women/2026-teams.csv` with `--women`):

```bash
mmnn nn predict <team1> <team2>
mmnn nn predict "Ohio State" TCU
```
Look in the appropriate `2026-teams.csv` for the correct team names to use.

## Installation

**From PyPI (end users):**

```console
pip install mmnn
```

**From source (development):**

Development uses [Hatch](https://hatch.pypa.io/) for environments and commands—not `pip install -e .`. Clone the repo and run the CLI or tests through Hatch:

```console
hatch run mmnn data fetch 2024
hatch run mmnn data process 2024
hatch run mmnn nn train
hatch run mmnn nn predict Duke UConn
hatch run test:test
```

`test:test` runs the `test` script in the `test` environment (pytest over `tests/`; see `[tool.hatch.envs.test]` in `pyproject.toml`). Use `hatch shell` if you want an interactive shell with the project and its dependencies on the path.

## License

`mmnn` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
