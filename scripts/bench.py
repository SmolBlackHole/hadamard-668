"""Benchmark and record solver performance across n values.

Records: date, commit, strategy, n, order, energy, iterations, elapsed, solved.
Appends to BENCHMARKS.md for tracking progress.
"""

from __future__ import annotations

import datetime
import subprocess
import time

import numpy as np

from builder import Builder
from solver import search as ils_search
from tracker import GramTracker


def bench_n(strategy: str, kind: str, n: int, steps: int, seed: int) -> dict:
    b = Builder(kind=kind, n=n)
    seqs = np.random.default_rng(seed).choice(
        (-1, 1), size=(b.k, n)).astype(np.int8)
    tracker = GramTracker(b.build)
    tracker.build(seqs, band_rows=b.band_rows, band_cols=b.band_cols)

    t0 = time.perf_counter()
    _best_seq, best_e, iters = ils_search(
        seqs, tracker, np.random.default_rng(seed), steps=steps)
    elapsed = time.perf_counter() - t0

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    return {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "commit": sha,
        "strategy": strategy,
        "n": n,
        "order": b.order,
        "energy": best_e,
        "iterations": iters,
        "elapsed": elapsed,
        "solved": best_e == 0,
    }


def main() -> None:
    configs: list[tuple[str, str, list[int], int]] = [
        ("gs4", "gs4", [8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], 200_000),
        ("golay_2n", "golay_2n", [6, 10, 12, 14, 18, 22, 26, 30, 34], 200_000),
    ]

    results = []
    for strategy, kind, ns, steps in configs:
        for n in ns:
            print(f"Benchmarking {strategy} n={n}...", end=" ", flush=True)
            r = bench_n(strategy, kind, n, steps, seed=42)
            status = "OK" if r["solved"] else f"e={r['energy']}"
            iters = r["iterations"]
            elapsed = r["elapsed"]
            print(f"{status}  iters={iters}  {elapsed:.1f}s")
            results.append(r)

    # write BENCHMARKS.md
    lines = ["# Benchmarks\n"]
    lines.append(
        "| Date | Commit | Strategy | n | Order | Solved | Energy | Iters | Elapsed |")
    lines.append(
        "|------|--------|----------|--:|-------|--------|--------|------:|--------:|")
    for r in results:
        solved = "YES" if r["solved"] else "NO"
        lines.append(
            f"| {r['date']} | {r['commit']} | {r['strategy']} | {r['n']} | "
            f"{r['order']} | {solved} | {r['energy']} | {r['iterations']} | {r['elapsed']:.1f}s |"
        )

    with open("BENCHMARKS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    solved_count = sum(1 for r in results if r["solved"])
    print(f"\nSolved: {solved_count}/{len(results)}")


if __name__ == "__main__":
    main()
