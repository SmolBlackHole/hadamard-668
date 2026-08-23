# Experiment data model

Stand: 2026-08-12

## State identity

Every stored GS4 state has two different identities:

- `validation_hash`: SHA-256 of the exact `(4,n)` `int8` payload. It protects
  `seqs_b64` against corruption and identifies one concrete state.
- `orbit_hash`: SHA-256 of the canonical representative under independent
  negashifts and reversals of the four sequences plus permutation of the four
  sequences.

`canonical_b64` stores that representative. `canonicalizer` is currently
`gs4-quick-orbit-v1`. Decimation is deliberately not part of this orbit. A
future larger equivalence relation must use a new version instead of changing
the meaning of this hash.

The original `seqs_b64` is never replaced by its canonical representative.
Legacy columns `sha256` and `class_hash` remain readable and mirror
`validation_hash` and `orbit_hash` for new records.

## Database layers

- `runs` is append-only experimental history. Reusing `(strategy,n,seed)` no
  longer deletes an older observation.
- `solutions` retains one row per exact `validation_hash`.
- `solution_features` stores recomputable metrics keyed by solution and feature
  version.

New runs record `candidate_budget`, actual `candidate_evals`, construction,
the complete solver configuration and the repository revision. Historical
`iterations` are not relabelled as candidate evaluations because their cost
semantics differed between solver versions.

Invalid historical rows are quarantined with `valid=0` and
`validation_error`; they are not deleted. The initial migration found one such
row, `solutions.id=101`, whose payload contains only zeros.

## Solution feature version 1

`gs4-solution-features-v1` records:

- sorted sequence sums and individual NAF signatures;
- all three pair-separator norms and complementary-pair membership;
- triple-product-dual energy and Strong-Split membership;
- basis and raw symbol distributions;
- the complete single-neighbour Q histogram;
- Quick-Orbit size and stabilizer;
- membership in the implemented Paley orbit.

Basis and symbol distributions describe the stored coordinate representative.
They are useful for solver dynamics but are not full orbit invariants.

## Solver state machine

With `--trace-phases`, each run emits compact events for:

```text
INITIALIZE -> GREEDY -> TABU -> TARGETED | RANDOM_KICK -> GREEDY
```

Each event records candidate cost, Q before/after, lowest Q, accepted downhill,
lateral and uphill moves, basis counts, exact endpoint hash, orbit endpoint hash
and outcome. Tabu is Markov only after its memory and RNG are included; the
phase trace is therefore an execution trace, not a claim that Q alone defines
the state.

## Reproduction

```powershell
python -m scripts.migrate_database data/hadamard.db
python -m scripts.analyze_solutions
python -m scripts.ablation --targeted --n 43 --n 47 --n 51 `
  --seeds 50 --candidate-budget 6000000 --workers 8 --trace-phases
```
