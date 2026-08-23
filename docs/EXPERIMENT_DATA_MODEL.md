# Experiment data model

Stand: 2026-08-23

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
the complete solver configuration and the repository revision. The required
legacy column `iterations` is written as zero for new runs. Historical values
remain untouched because their cost semantics differed between solver versions.

## Search metrics version 2

`stats_json.stats_schema_version=2` separates three questions which the old
`singles`/`kicks` counters mixed together:

1. Did the run solve, and from which phase and Q level?
2. How much wall time and candidate work did it consume?
3. Which accepted state transitions produced that work?

Candidate work is stored as `greedy_candidate_evals`,
`tabu_candidate_evals`, `escape_candidate_evals` and
`quench_candidate_evals`. Their sum is `total_candidate_evals` and must equal
the charged budget. These are logical solver work units, not homogeneous CPU
instructions. They support paired comparisons within the same solver revision,
but not CPU/GPU or cross-implementation performance claims.

Accepted transitions are stored separately as `greedy_moves`, `tabu_moves`,
`random_kicks`, `targeted_quenches` and `quench_moves`. A targeted quench is one
branched start plus all transitions in its nested search. It is not also counted
as a random kick. `total_accepted_moves` is useful for explaining the search
path, but it is not a performance score.

`greedy_time_s`, `tabu_time_s`, `escape_time_s` and `rebuild_time_s` measure the
top-level wall-time attribution. Escape time includes nested quench execution;
nested time is not added again to the other phase timers.

The ablation report makes these primary:

- solve rate over fixed paired seeds;
- total wall seconds per solution, including failed-run cost;
- solutions and unique Quick-Orbits per worker-hour;
- median and p90 time and candidate evaluations among solved runs;
- exact solution and Quick-Orbit counts under the recorded canonicalizer.

Each worker executes the same untimed 10,000-evaluation warm-up before its first
measured run. This prevents Numba compilation from being charged only to the
first configuration. `solutions_per_worker_hour` is therefore steady-state
single-worker-equivalent throughput. The report's overall elapsed time still
includes process startup and warm-up and is not used for that rate.

The conditional median and p90 explain run-to-run variance. They must always be
read together with solve rate and total cost per solution, otherwise failed runs
disappear from the result.

## Cross-solver comparison contract

Before comparing this solver with another pipeline, record the sequence family
(negacyclic or cyclic), `n` and matrix order, start distribution, seed set,
stopping rule, code revision, hardware, worker count, exact verification and
the equivalence relation used for deduplication.

PatternBoost searches Goethals-Seidel arrays built from cyclic matrices using a
Fourier local-improvement stage plus transformer generations. This solver
searches a negacyclic GS4 family. Results from those two spaces belong in
separate benchmark rows. Candidate evaluations are implementation-specific;
only fully specified wall-time yield and unique verified output can be compared,
and the structural difference must remain visible. See the
[PatternBoost paper](https://arxiv.org/abs/2604.11101).

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
the state. A targeted event counts branched starts and nested transitions in
`accepted_moves`; its direction buckets remain zero because those branches do
not form one top-level trajectory.

## Reproduction

```powershell
python -m scripts.migrate_database data/hadamard.db
python -m scripts.analyze_solutions
python -m scripts.ablation --targeted --n 43 --n 47 --n 51 `
  --seeds 50 --candidate-budget 6000000 --workers 8 --trace-phases
```
