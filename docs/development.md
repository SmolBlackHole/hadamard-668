# Development

Parent: [Documentation index](README.md)

## Contents

- [Development](#development)
  - [Contents](#contents)
  - [Environment](#environment)
  - [Quality checks](#quality-checks)
  - [Build distributions](#build-distributions)
  - [Continuous integration](#continuous-integration)
  - [Research utilities](#research-utilities)

## Environment

Run commands from the repository root. The project requires Python 3.11 or newer;
NumPy and Numba provide the numerical runtime. The [README](../README.md#quick-start)
shows a standard virtual environment and pip installation.

For the dependency versions recorded in `uv.lock`, use uv:

```bash
uv sync --locked --extra dev --python 3.13
uv run --locked --extra dev python scripts/quality.py check
```

Use a separate environment if you already have an active environment with another
Python version. To update dependencies deliberately, run `uv lock`, review the
lockfile diff and rerun the quality checks. Do not discard the lockfile to resolve
an installation failure.

VS Code recommendations are optional. The workspace points at `.venv`, allowing
the Python extension to resolve the platform-specific interpreter path.

Install development dependencies in the environment selected by your editor, too:

```bash
python -m pip install -e ".[dev]"
python -m pyright
```

An environment containing only runtime dependencies cannot resolve test helpers
such as `pytest.raises` or the documentation tooling. This can produce many unknown
type warnings even when checks in a separate environment pass. Select
`.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Linux and macOS.
After installing missing packages, restart the editor's language server if stale
diagnostics remain. Keep the strict checks enabled.

## Quality checks

With the development environment activated:

```bash
python scripts/quality.py check
```

The command runs documentation checks, Ruff formatting and lint checks, strict Pyright, pytest and
`compileall`, stopping at the first failure. Its authoritative command list is
[`scripts/quality.py`](../scripts/quality.py). Test output stays under `runs/`.

Useful focused commands:

```bash
python scripts/quality.py docs
python -m pytest tests/test_verify.py tests/test_pipeline.py -q
python scripts/quality.py coverage
python scripts/quality.py mutate
```

Mutation testing is optional and can take substantially longer than the standard
checks. `python scripts/quality.py fix` applies formatting and automatic lint fixes;
review its diff before committing.

## Build distributions

```bash
uv build
```

This produces a source archive and wheel under ignored `dist/`. Packaging explicitly
includes the existing `src` package and `run` module. The command examples use a
repository checkout; research scripts and documentation are maintained there.

## Continuous integration

The [quality workflow](../.github/workflows/quality.yml) runs the locked dependency
installation, standard checks, distribution build and an exact order-208 construction
on Windows and Linux with Python 3.11 and 3.13. Local checks do not establish that
the remote jobs have passed; inspect their actual results on GitHub.

## Research utilities

| Command | Purpose |
| --- | --- |
| `python -m lab.compare --help` | Paired comparisons of solver configurations |
| `python -m lab.catalog --help` | Generate a catalog from an existing database |
| `python -m scripts.migrate_database --help` | Migrate stored identities and audit the result |
| `python -m lab.diagnose --help` | Diagnose the local Q=1 landscape |
| `python -m lab.compare --compare-starts --help` | Compare random and cyclic starts |
| `python -m src.interleaver --check` | Sample the alternation identity |
| `python -m src.deinterleaver --verify` | Check experimental deinterleaving round trips |

The [laboratory guide](laboratory.md) describes bounded comparisons and recovery experiments.
The interleaver and deinterleaver are standalone research programs, with their own
construction implementations. They do not define the production acceptance path.
Their numerical experiments are not proofs of the accompanying mathematical claims.
