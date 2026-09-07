# Solver research worklist

Parent: [Documentation index](README.md)

This is the execution worklist for the authorized research round. Experimental
changes remain opt-in until independently confirmed. The mathematical target,
exact acceptance checks and production defaults stay fixed during comparisons.

## Protocol

- Preserve recovered history under ignored `research-archive/`, with source
  commit IDs. Historical claims are leads, not current validation.
- Store raw reports under `runs/research/`; keep commands and compact conclusions
  in documentation. Record configuration, seed range, budget, code and environment.
- Screen on paired random starts at n=40,46,50 with 300 seeds and 6M evaluations.
  n=40 provides a less sparse signal; n=46,50 test the difficult regime.
- Confirm the strongest candidate, if any, on disjoint seeds 1042..2041 at
  n=40,46,50,54. Include baseline and change only the tested factor.
- Compare solve yield and total worker time including failures. Charge added
  candidate scoring to the budget. Keep timing benchmarks free of parallel tests.
- Use exact independent verification for solutions and label restricted symmetry
  groups accurately. Do not claim full Hadamard equivalence from GS4 orbit hashes.
- Stop an unpromising experiment after the screen. Promote no default based on
  a handful of wins or a lower intermediate Q. Retain useful research controls
  separately from default runtime behavior.

## TODO

- [x] R1: Recover historical research documents, provenance and prior negative results.
  Recovered 33 exact Markdown blobs (300,699 bytes); local index:
  `research-archive/README.md`. Includes mature and original research snapshots.
- [x] R2: Reproduce the existing quench-allocation result and record it as prior
  evidence (already measured: 12,000 runs, no better arm). Avoid rerunning it.
- [x] R3: Test removal of the low-Q greedy handoff (`qwindow_high=0`) against 9.
- [x] R4: Test tabu memory duration (decay 0.95 versus 0.7) and a shorter walk
  (100 versus 400 steps), with nested targeted search disabled in all arms so
  outer configuration cannot be silently reset by nested defaults.
- [x] R5: Test targeted candidate positional bias: random matching-column samples
  versus the first five matching columns, with a shared 100k quench limit.
- [x] R6: Re-test complete-residual ranking as a negative control for the recovered
  historical result. Charge ranking work and compare to unranked random proposals.
- [x] R7: Diagnose Q=1 endpoints with exact single/pair, distinct-sequence triple,
  and one-flip-per-sequence quadruple repair searches. Record completeness limits.
- [x] R8: Add and validate common negacyclic decimation classification composed
  with the quick orbit. Compare solution counts under both named groups. Assess
  whether unrestricted matrix equivalence is needed for interpreting this round.
- [x] R8b: Test equal-Q tabu snapshot adoption, a recovered unresolved hypothesis.
  The first eight-arm screen found no convincing winner. Compare the equal-Q flag
  with both targeted escape enabled and disabled; preserve strict-improvement metrics.
- [x] R7b: Exhaust all remaining three-flip repairs, including repeated rows.
  All 162 endpoints tested, no repairs; 42.16 seconds including solution classification.
- [x] R9: Independently confirm a promising screen winner, or explicitly record
  no winner. Investigate whether solve gains are confined to easier lengths.
  No convincing screen winner; no holdout or default promotion justified.
- [x] R10: Summarize results and mathematical implications; run repository quality
  checks, review scope, and leave no unexplained failing check or experiment.
  [Results](research-results.md) cover 10,800 new search runs and exact diagnostics.
  Full quality check: 157 tests passed, Ruff clean, Pyright without errors or warnings.
  GitHub math rendering corrections are part of this research branch's documentation.

## Conditional follow-ups

These are evaluated after the bounded screen, not unlimited commitments:

- If candidate ranking loses, inspect basin reachability rather than inventing
  another monotone energy score.
- If tabu memory helps, separately test persistence across walks and equal-Q
  endpoint adoption. Do not combine these changes before attribution is possible.
- If equivalent solution counts collapse under decimation, retain the expanded
  group as an analysis tool; do not change database identities implicitly.
- Full matrix equivalence requires a separate exact algorithm and validation,
  and is only warranted if restricted identities obscure the measured conclusion.

No conditional follow-up is active after this round. Equal-Q retention was tested
separately because it was an explicit unresolved lead in the recovered notes.
Memory persistence and unrestricted quadruples remain future questions, not
unfinished acceptance checks for this round.

## Empirical landscape round

Authorized follow-up: use existing data first, retain reproducible prototypes,
and make a direct production change only if the evidence supports it.

- [x] E1: Explain the paired n=56 master/research discrepancy using identical
  starts, seeds and budgets; distinguish stored observations from reproduction.
- [x] E2: Compare perturbed verified solutions against stored failed states at
  matched n and Q. Record parent solution, perturbation and local move geometry.
- [x] E3: Run equal-budget continuations from both groups; measure solve yield
  and evaluate whether geometry separates known-near from search-selected states.
- [x] E4: Document results, limitations and the next justified experiment. Keep
  negative results, avoid adding another production policy switch.
- [x] E5: Verify analysis code and generated artifacts with focused checks.
  Full quality check: 175 tests pass; Ruff and Pyright clean. Recovery plot
  visually inspected. All 450 direct production replays match the prototype.
- [x] E3b: Diagnose one-bit recovery failures and test full scanning without a
  production switch. 220/220 controls solve, but zero additional random-start
  solves in the 900-run screen. Subsequently integrate directly at the user's
  request and replay the 450 full-scan runs through the production solver.

Results and the corrected 6M-versus-10M comparison are in
[Landscape investigation](landscape-results.md). The five matched pairs do not
support a classifier; this acceptance criterion is closed as insufficient evidence.

## Laboratory consolidation

- [x] Move reusable tools to `lab/` and completed drivers to `lab/experiments/`.
- [x] Extract configurable recovery with explicit parents and shared pair diagnostics.
- [x] Replace the fixed-file start benchmark with retained paired comparisons.
- [x] Snapshot source for new comparisons/recovery and generate plots automatically.
- [x] Update commands, test imports, packaging and quality-check coverage.
- [x] Finish focused CLI smoke tests and full quality checks before committing.
  Full quality suite: 175 tests; two additional laboratory CLI tests pass.
  Multi-process start comparison, real-database recovery, source distribution
  and wheel build pass. Ruff and Pyright are clean.

The next architecture proposal covers modular solver phases and consolidation of
the standalone interleaver/deinterleaver programs. No solver phase changes belong
to the laboratory consolidation itself.

## Approved solver architecture refactor

- [x] A1: Capture deterministic states, budgets, statistics and warm timings.
- [x] A2: Split solver responsibilities, define snapshot ownership and explicit phase injection.
- [x] A3: Migrate laboratory callers and consolidate interleaving using canonical helpers.
- [x] A4: Verify replay, tests, package contents and warm runtime before closing the refactor.

Preserve algorithm behavior, RNG consumption and budget rules. No algorithm
experiment or automatic commit is part of this architecture change.

Validation against the pre-refactor baseline `aa3bab4`: 56 searches at n=13/36/56/167,
seven configurations and two seeds reproduce sequence bytes, energy, candidate
counts and all non-timing statistics exactly. The archived n=56 comparison
reproduces all 32 paired seeds, and recovery reproduces all 1,100 stored records.
Local evidence is retained under the ignored `runs/refactor/` directory.

The full quality check passes 191 tests, documentation checks, Ruff, Pyright and
compilation. The wheel includes all eight solver modules and runs a search from
an isolated extraction. A two-worker laboratory comparison also succeeds and
captures the complete solver package in its source snapshot. Both interleaving
lifts reproduce the former implementations and their folds invert bitwise.

Warm n=56 timing, 20 seeds at 6M evaluations per repetition: baseline median
1.671s, post-refactor median 1.745s (+4.4%). Three repetitions per version have
overlapping ranges (1.392-1.726s versus 1.568-1.783s). This small timing sample
does not establish a speedup or isolate the refactor's runtime cost.
