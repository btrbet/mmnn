# mmnn

[![PyPI - Version](https://img.shields.io/pypi/v/mmnn.svg)](https://pypi.org/project/mmnn)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmnn.svg)](https://pypi.org/project/mmnn)

-----

This project creates NCAA Division I college basketball tournament bracket predictions using a neural network (men's or women's).

This project was inspired by [this paper](https://github.com/btrbet/mmnn/blob/main/research/Comparing%20Various%20Machine%20Learning%20Statistical%20Methods%20Using%20Vari.pdf).

Tournament data for 2010–2026 (men's) is included under `data/men/`. To add more years, use `mmnn data fetch`. Use `-w` / `--women` on `mmnn data …` and `mmnn nn …` to use women's data under `data/women/` (Sports Reference URLs use `women` instead of `men` in the path).

## Usage

**Basic workflow (men's, default):** Processed `*-data.csv` files are already in `data/men/`, so you can train immediately. To add or refresh a year, run `mmnn data fetch <year>` and `mmnn data process <year>` first.

```bash
mmnn nn train
mmnn nn bracket 2025
mmnn nn predict Duke Siena
```

**Women's tournament** — add `-w` or `--women` to each command below. Bracket evaluation needs at least two processed years total (e.g. another year already in `data/women/` plus the year you evaluate):

```bash
mmnn data fetch 2025 --women
mmnn data process 2025 --women
mmnn nn train --women
mmnn nn bracket 2025 --women
mmnn nn predict "South Carolina" UConn --women
```

### Fetch data

```bash
mmnn data fetch <year>
mmnn data fetch <year> --women
```

Fetch the raw bracket and team stats for the given year from [Sports Reference](https://www.sports-reference.com/cbb/).

### Process data

Process raw data for a given year into the format needed by the neural network. Reads `data/men|women/YEAR-teams.csv` and `YEAR-games.csv`, then writes `YEAR-data.csv` with per-game delta features and a Winner label.

```bash
mmnn data process <year>
mmnn data process <year> --women
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

### Evaluate a full bracket (holdout year)

Retrain the network on every `*-data.csv` **except** the bracket year, then predict each game in that year’s tournament and print per-game results plus accuracy, log loss, and related metrics. The model is fit in memory only; **it does not read or overwrite** `data/men/model.pt` or `data/women/model.pt`.

You need at least one other processed year besides the bracket year (`mmnn data process <year>`), and that year’s `{year}-games.csv` and `{year}-teams.csv` must exist.

```bash
mmnn nn bracket 2025
mmnn nn bracket 2025 --women
```

Optional **`--epochs`** sets the training epoch count (same default as `mmnn nn train`). Useful for quicker runs while iterating:

```bash
mmnn nn bracket 2025 --epochs 50
```

### Predict a single matchup

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
hatch run mmnn nn bracket 2025
hatch run mmnn nn predict Duke UConn
hatch run test:test
```

`test:test` runs the `test` script in the `test` environment (pytest over `tests/`; see `[tool.hatch.envs.test]` in `pyproject.toml`). Use `hatch shell` if you want an interactive shell with the project and its dependencies on the path.

## License

`mmnn` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
