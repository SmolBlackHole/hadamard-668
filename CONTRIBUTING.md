# Contributing

Parent: [Project README](README.md)

Use the [development guide](docs/development.md) for setup and checks. Keep changes
focused and describe the problem, resulting behavior and verification in the pull
request. Discuss changes to the mathematical model or public CLI before implementing
a broad redesign.

Run `python scripts/quality.py check` before submitting. Report failed checks,
skipped platforms and remaining uncertainty. Changes to energy, construction or
acceptance need regression evidence against independent verification.

Performance claims need paired seeds, the same work budget and retained raw
results. Follow the [experiment contract](docs/experiments.md); a single fast
solve is not a benchmark.

Write documentation and code comments in English. Follow the
[documentation guide](docs/writing-and-maintaining-docs.md). Keep generated
databases, benchmark output, credentials and personal editor settings out of commits.

Contributions are made under the repository's [MPL-2.0 license](LICENSE).
