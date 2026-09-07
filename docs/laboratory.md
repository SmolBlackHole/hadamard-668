# Research laboratory

Parent: [Documentation index](README.md)

## Contents

- [Responsibilities](#responsibilities)
- [Commands](#commands)
- [Experiment contract](#experiment-contract)
- [Completed experiments](#completed-experiments)
- [Algorithm changes](#algorithm-changes)

## Responsibilities

`src/` owns the production solver, exact constructions, verification and persistence.
`run.py --sweep` remains the normal search entry point. `lab/` measures and compares
that implementation. `scripts/` contains quality checks and database maintenance.
The standalone interleaver/deinterleaver programs still live in `src/`; their
consolidation belongs to the upcoming architecture change.

| Module | Responsibility |
| --- | --- |
| `lab.compare` | Paired configuration or start comparisons with raw results and plots |
| `lab.plot` | Read-only database cohort plots, CSV and provenance |
| `lab.recover` | Controlled perturbations of explicit verified parent solutions |
| `lab.diagnose` | Q=1 single/pair analysis; optional extended repair witnesses |
| `lab.neighborhoods` | Exact larger neighborhoods and specified-group classification |
| `lab.catalog` | Versioned solution catalog |
| `lab.provenance` | Source snapshots for new comparison and recovery experiments |

Recovery uses the same exact pair diagnostic as Q=1 analysis. The larger
neighborhood engine retains its pair-sum maps for triple and restricted quadruple
searches; those maps serve a different task from counting all pair repairs.

## Commands

Install `python -m pip install -e ".[dev]"` and run from the repository root:

```bash
python -m lab.plot --database data/hadamard.db --output runs/plots
python -m lab.compare --compare-starts --n 36 40 --seeds 100 --seed-start 1042 --candidate-budget 6000000 --workers 4 --output runs/start-comparison.json
python -m lab.recover --help
python -m lab.diagnose --input data/hadamard.db --n 52 --output runs/q1-52.json
python -m lab.catalog --database data/hadamard.db --output runs/catalog.json
```

Recovery requires `--parent-ids` identifying solved, validated rows in `runs`, not
IDs in the `solutions` table. Supply `--radii`, `--repeats`, `--seed`,
`--candidate-budget` and a fresh `--output` JSON path. Parent sequences are verified
independently and retained in the report, together with perturbations and seeds.
Each parent gets its own recovery curve; repeated perturbations are dependent.

`lab.diagnose --extended` additionally searches all triples and quadruples with
one flip per sequence. It does not exhaust unrestricted quadruples and can be
expensive. `lab.neighborhoods --first-run-id ... --last-run-id ...` provides the
combined endpoint and classification report used by the earlier experiments.

Comparison and recovery generate JSON, PNG and SVG. Database plotting also writes
CSV summaries and a Markdown index. The removed `bench_starts.py` used fixed
temporary database names and deleted them afterward; `--compare-starts` instead
retains the raw results and never uses those temporary databases.

## Experiment contract

Before running, state the question, baseline, budgets and acceptance criterion.
Use identical seed sets for paired comparisons and separate confirmation seeds
after selecting a candidate. Count all scoring work and retain unsuccessful runs.

New comparison and recovery reports require a fresh output path and save an
adjacent source directory with hashes and runtime package versions. Keep source
unchanged while measuring. Source snapshots do not replace retained input data;
dirty Git labels alone do not identify the executed code.

The database remains unchanged during analysis. Different budgets, configurations,
revisions and odd/even lengths must not be silently pooled. Legacy experiment
drivers have their own documented fixed batches and output locations.

## Completed experiments

The drivers in [lab/experiments](../lab/experiments/README.md) reproduce completed
investigations. Reusable tools must not import them or require historical Git
versions. Their fixed seeds and batches are part of the archived experiment,
not defaults for new laboratory work.

## Algorithm changes

Test a bounded prototype first. A useful result must improve the intended task,
not only an easy synthetic control or an intermediate Q value. Only then consider
optimization and direct replacement of an existing production algorithm.
Do not accumulate production switches for every abandoned prototype.

Keep refactoring separate from algorithm changes: first preserve states, budget
counts and seeded behavior; then measure a proposed behavioral change explicitly.
