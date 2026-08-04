"""Head-to-head comparison of ALL repair strategies.

Usage::

    PYTHONPATH=src python -m experiments.bench_strategies [--tt 8,16] [--trials 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from correlations import (  # noqa: E402
    TURYN_WEIGHTS,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
)
from experiments.strategies import (  # noqa: E402
    HybridRepair,
    PhaseRetrievalRepair,
)
from experiments.syndrome_repair import RepairExperiment  # noqa: E402
from fixtures import tt_lengths, tt_sequences  # noqa: E402
from strategies.kflip import KFlipRepair  # noqa: E402


def _make_strategies(n: int):
    """Build all strategies with sensible defaults for TT(n)."""
    return [
        (
            "repair-exp",
            RepairExperiment(
                n=n,
                sieve=False,
                pair_top=64,
                triple_top=20,
                plateau_threshold=8,
                plateau_noise=4,
                verbose=False,
            ),
        ),
        (
            "repair-metro",
            RepairExperiment(
                n=n,
                sieve=False,
                pair_top=64,
                triple_top=20,
                plateau_threshold=8,
                plateau_noise=4,
                temperature=80.0,
                cooling=0.995,
                verbose=False,
            ),
        ),
        ("kflip", KFlipRepair(n=n, sieve=False, max_k=5)),
        ("phase-ret", PhaseRetrievalRepair(n=n, sieve=False, hio_iters=200, beta=0.9)),
        ("hybrid", HybridRepair(n=n, sieve=False, max_k=5, hio_iters=50)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tt", default="8,16")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(_REPO / "results"))
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.tt.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    d_values = [1, 2, 3, 4, 5, 7, 10]
    rng = np.random.default_rng(args.seed)

    for n in n_values:
        sol = tt_sequences(n)
        lengths = tt_lengths(n)
        B = int(lengths.sum())
        all_pos = [(r, c) for r in range(4) for c in range(int(lengths[r]))]

        strategies = _make_strategies(n)
        out = out_dir / f"bench_TT{n}.jsonl"

        print(f"\n{'=' * 70}")
        print(
            f"  TT({n}) — {B} bits — {args.trials} trials x {len(d_values)} distances x {len(strategies)} strategies"
        )
        print(f"{'=' * 70}")
        print(f"  {'d':>3} {'strategy':>12} {'success':>10} {'avg_e':>10} {'avg_t':>8}")
        print(f"  {'-' * 48}")

        for d in d_values:
            if d > B // 2:
                continue
            step_budget = max(120, 30 * d)
            for s_name, strategy in strategies:
                success = 0
                total_e = 0
                total_t = 0.0
                for trial in range(args.trials):
                    damaged = sol.copy()
                    chosen = rng.choice(B, size=d, replace=False)
                    for idx in chosen:
                        r, c = all_pos[int(idx)]
                        damaged[r, c] *= -1
                    e0 = float(
                        nonperiodic_correlation_energy(
                            nonperiodic_autocorrelation_state(
                                damaged, lengths=lengths, weights=TURYN_WEIGHTS
                            )
                        )
                    )

                    t0 = time.perf_counter()
                    result = strategy.search(
                        steps=step_budget, seed=args.seed + d * 100 + trial, sequences=damaged
                    )
                    wall = time.perf_counter() - t0
                    total_t += wall
                    ee = result.metrics["energy"]
                    total_e += ee
                    if ee == 0:
                        success += 1

                    row = {
                        "n": n,
                        "d": d,
                        "trial": trial,
                        "strategy": s_name,
                        "start_energy": e0,
                        "end_energy": ee,
                        "success": ee == 0,
                        "wall_s": wall,
                    }
                    with open(out, "a") as f:
                        f.write(json.dumps(row) + "\n")

                avg_e = total_e / args.trials
                avg_t = total_t / args.trials
                print(
                    f"  {d:>3} {s_name:>12} {success:>8}/{args.trials} {avg_e:>10.0f} {avg_t:>7.1f}s"
                )

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
