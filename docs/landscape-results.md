# Empirical landscape investigation

Parent: [Documentation index](README.md)

## Contents

- [Decision](#decision)
- [Budget correction](#budget-correction)
- [Matched states](#matched-states)
- [Recovery experiments](#recovery-experiments)
- [Full-scan prototype](#full-scan-prototype)
- [Reproduction and limits](#reproduction-and-limits)
- [Next question](#next-question)

## Decision

At the user's request, full scanning is now implemented directly in the research
branch's production solver, without a policy switch. It repairs a demonstrable
first-improvement weakness on one-bit perturbations, but produces no additional
solves in the paired random-start screen. It is not a demonstrated scaling breakthrough.
Low-Q matching provides
too few controls to establish a structural predictor. Both are useful negative
results, not evidence that all alternative neighborhoods fail.

## Budget correction

The two sweeps did not have equal budgets. Run IDs 6301..20300 use 6M candidate
evaluations; IDs 20301..28300 use 10M. The later n=167 batch also uses 10M.
The earlier interpretation that the n=56 discrepancy indicated changed default
solver behavior was incorrect: it overlooked the stored budget field.

Replaying n=56 seeds 42..73 with master solver source at `bb26d1d` and the current
research implementation gives identical output bytes at 6M. All 32 reproduce the
first sweep. Replaying research at 10M reproduces all 32 second-sweep records.
This resolves the sampled discrepancy; it is not an exhaustive equivalence proof.

## Matched states

Use only the 6M batch and n=36,40,44,48,52. Select at most three distinct exact
solution records per length (only one available at n=48/52), independently verify
each parent, and generate 800 perturbations per parent, 8,800 total. Perturbations
flip 1,2,4 or 8 distinct bits uniformly. Half also undergo at most 64 steepest
improving flips. Record exact distance to the parent after descent; this is an
upper bound on distance to any solution, not a symmetry-minimized distance.

After removing solved controls and exact duplicates, match against failed stored
states at identical n and Q. Only five pairs exist: one each at n=36,40,44, none
at n=48, and two at n=52. These are exhausted available matches, not a representative
sample of all possible near-solution states.

Every matched state lacks improving or neutral single flips. Exhaustive distinct
two-flip endpoint scoring finds no direct repair in either group. Three fresh
250k continuations per state give zero solves in all 30 runs. This sample cannot
support a useful prediction claim; repeated continuations are not independent states.

## Recovery experiments

Generate a separate set of perturbations from the same verified parents with RNG
seed 20260908, 20 per parent per radius. Run one 250k continuation each, with RNG
seeds 300000..300019 reused across parents and radii. Total: 1,100 search runs.
The perturbation radius is a certified upper bound on solution distance.

| n | Trials per radius | 1 bit | 4 bits | 8 bits | 16 bits | 32 bits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 36 | 60 | 39 | 0 | 2 | 2 | 1 |
| 40 | 60 | 38 | 0 | 3 | 2 | 0 |
| 44 | 60 | 42 | 0 | 0 | 0 | 0 |
| 48 | 20 | 12 | 2 | 0 | 0 | 0 |
| 52 | 20 | 13 | 0 | 0 | 0 | 0 |

Cells count verified solutions. These small, parent-dependent counts are diagnostic;
they do not estimate global basin volume. In particular, apparent nonmonotonicity
with radius is not established as a general effect.

Only 144/220 one-bit controls solve, despite having an immediately solving flip.
First-improvement greedy can accept another improvement before checking that flip.
This is a limitation of the heuristic selection rule, not an incorrect energy.

## Full-scan prototype

The prototype scores every affordable single before selecting a move. A direct
solution takes priority; otherwise it keeps the first-improvement ordering.
It charges every scored candidate, including scores after the first improvement,
and changes no production configuration. The default geometric weight must be zero.

Replaying the 220 one-bit controls solves all 220. The largest consumed budget is
208 evaluations, consistent with a complete scan at n=52 (initialization is free).

The paired screen uses n=36,40,44, seeds 42..191 and 6M evaluations, 900 runs:

| n | Baseline solves / 150 | Full-scan solves / 150 |
| --- | ---: | ---: |
| 36 | 79 | 79 |
| 40 | 14 | 14 |
| 44 | 0 | 0 |

There are no discordant solved seeds. Best Q is identical except for one n=44
seed where full scan is worse. The screen did not justify automatic promotion;
the subsequent direct integration is an explicitly requested experiment.
This does not establish equivalence at every length or rule out cheaper checks.

Direct integration replayed all 450 full-scan screen runs through the production
solver. Every solved flag, final Q and candidate count matches the prototype.

## Reproduction and limits

```bash
python -m lab.experiments.replay
python -m lab.experiments.landscape
python -m lab.experiments.greedy
```

These are bounded research drivers with explicit batch IDs, seeds and budgets in
source. They read the database and write JSON plus PNG/SVG under
`runs/research/landscape/`. The greedy prototype temporarily replaces the function
inside its own process and restores it on exit; it does not edit production code.
The replay and recovery drivers explicitly reload the original scan from `bb26d1d`
so the recorded baseline remains reproducible after direct integration.
Do not run it as a benchmark concurrently with unrelated searches. Timings are
retained but no performance claim is made from this screen.

Artifacts: `replay.json`, `matched.json`, `recovery.json`, `greedy-screen.json`,
`greedy-recovery.json` and `recovery.png`/`recovery.svg`. Parent IDs and perturbation
positions reconstruct recovery inputs; matched states are stored as exact bytes.
The database and research source must be retained together. Dirty Git revisions
alone cannot reproduce uncommitted source. `source-manifest.json` records the exact
source hashes for this round.

All observed solves pass independent residual and full-matrix verification.
The neighborhood/scan helpers have focused regression tests. Synthetic controls
are not naturally sampled states, and the investigation contains no known n=167
solution from which to construct controls.

## Next question

The next useful experiment is whether multi-bit proposals can recover deliberately
damaged solutions while retaining enough efficiency to help random starts. Use the
same parent-controlled recovery protocol before a paired random-start screen.
Historical symbol and pair rescues were often negative, so a new proposal needs
a specific structural distinction (for example coordinated cross-column changes).
Do not train a state classifier from the five matched pairs or scale the one-bit
control success into a claim about order 668.
