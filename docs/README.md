# Documentation

Parent: [Project README](../README.md)

This documentation describes the current implementation. Claims about experiments
require retained data; historical aggregates without their source artifacts are
not reproducible evidence.

## Contents

- [Documentation](#documentation)
  - [Contents](#contents)
  - [Reading guide](#reading-guide)
  - [Sources of truth](#sources-of-truth)
  - [Notation](#notation)
  - [Maintaining documentation](#maintaining-documentation)

## Reading guide

| Question | Read |
| --- | --- |
| What mathematical problem does the solver search? | [Mathematics](mathematics.md) |
| How does the search work? | [Solver](solver.md) |
| When can a matrix be constructed directly? | [Constructions](constructions.md) |
| How do I store, audit and compare results? | [Experiments](experiments.md) |
| Which module owns each part of the system? | [Architecture](architecture.md) |
| How do I install and check the project? | [Development](development.md) |
| How should documentation be maintained? | [Writing documentation](writing-and-maintaining-docs.md) |

## Sources of truth

Code defines implemented behavior; tests provide executable regression evidence.
The [architecture guide](architecture.md#ownership) maps responsibilities to modules.
Mathematical derivations state their assumptions and link to the relevant tests.

Local databases and files under `runs/` are not shipped with the repository.
Reproduce and retain your own experiment artifacts as described in
[Experiments](experiments.md). A lower energy, a faster single run or more accepted
moves alone does not demonstrate a better solver.

## Notation

`n` is the length of each of the four binary sequences. The matrix order is `4n`.
`r` is the combined negaperiodic residual, `u = r/4` its reduced form over
independent lags, `Q = ||u||²` the search objective, and `E = 64nQ` the repository's
Gram energy. See [Mathematics](mathematics.md) for the derivation and valid domain.

## Maintaining documentation

Give each fact one authoritative page and link to it from related topics.
Follow the [documentation guide](writing-and-maintaining-docs.md) when adding,
translating or moving a page.
