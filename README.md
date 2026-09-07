# Hadamard-668

[![Quality](https://github.com/SmolBlackHole/hadamard-668/actions/workflows/quality.yml/badge.svg)](https://github.com/SmolBlackHole/hadamard-668/actions/workflows/quality.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MPL-2.0](https://img.shields.io/badge/license-MPL--2.0-blue)](LICENSE)

CPU-based research software for finding real Hadamard matrices in the negacyclic
Goethals-Seidel family. The long-term target is order 668: four binary sequences
of length `n = 167`.

**This project has not found a matrix of order 668.** It provides a heuristic
solver for smaller GS4 instances, exact Paley/Ito constructions, independent
verification, and SQLite-backed experiments. A search can finish without a solution.

## Quick start

Clone the repository and install it with Python 3.11 or newer:

```bash
git clone https://github.com/SmolBlackHole/hadamard-668.git
cd hadamard-668
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in bash, then run:

```bash
python -m pip install -e ".[dev]"
python run.py --strategy paley-ng --order 208 --no-output
```

This deterministic construction builds and verifies an order-208 matrix. For
a seeded heuristic search:

```bash
python run.py --strategy gs4 --order 128 --candidate-budget 6000000 --seed 42 --no-output
```

`--order` is the matrix order `4n`; `--sweep` takes sequence lengths `n`.
Omit `--no-output` to store runs in the local `data/hadamard.db` database.
Numba compilation adds overhead to the first search in a process.

## Documentation

Start at the [documentation index](docs/README.md):

- [Mathematics](docs/mathematics.md): GS4 equations, energy and flip algebra.
- [Solver](docs/solver.md): search phases, settings and candidate budgets.
- [Constructions](docs/constructions.md): exact paths and their limits.
- [Experiments](docs/experiments.md): persistence, analysis and reproducibility.
- [Development](docs/development.md): installation, checks and contribution workflow.

The recursive `construct` strategy has a known
[budget accounting limitation](docs/constructions.md#recursive-budget-limitation).

## Contributing

Run `python scripts/quality.py check` before proposing changes. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the contribution contract and
[SECURITY.md](SECURITY.md) for reporting security issues.

Citation metadata is available in [CITATION.cff](CITATION.cff). When reporting
experimental results, include the commit, configuration, seeds and budget.

## License

This Source Code Form is subject to the terms of the Mozilla Public License,
v. 2.0. See [LICENSE](LICENSE) for the complete license text.
