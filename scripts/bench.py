"""Benchmark and record solver performance across n values.

Records: date, commit, strategy, n, order, energy, iterations, elapsed, solved.
Appends to BENCHMARKS.md for tracking progress.
"""

from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path


def bench_n(
    strategy: str, kind: str, n: int, steps: int, seed: int, dataset: str | None = None
) -> dict:
    """Benchmark via Generator.search() (CLI path). Optionally save to dataset."""
    from generator import Generator

    gen = Generator(kind=kind, n=n)
    r = gen.search(steps=steps, seed=seed)

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    if r.metrics.energy == 0 and dataset:
        from output import append_solution

        append_solution(
            Path(dataset), strategy, n, seed, r.sequences, r.elapsed, r.iterations, r.stats
        )

    return {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "commit": sha,
        "strategy": strategy,
        "n": n,
        "order": gen.order,
        "energy": r.metrics.energy,
        "iterations": r.iterations,
        "elapsed": r.elapsed,
        "solved": r.metrics.energy == 0,
    }


def _fmt_time(t: float) -> str:
    if t < 0.001:
        return f"{t * 1_000_000:.0f}us"
    if t < 1.0:
        return f"{t * 1000:.0f}ms"
    return f"{t:.1f}s"


def main() -> None:
    args = sys.argv[1:]

    # --sweep gs4 24 28 30 --seeds 50   ->  n=24,28,30 each 50 seeds
    if "--sweep" in args:
        idx = args.index("--sweep")
        strategy = args[idx + 1]
        # collect n values until we hit --seeds or --dataset
        ns: list[int] = []
        num_seeds = 50
        dataset = None
        i = idx + 2
        while i < len(args):
            if args[i] == "--seeds":
                num_seeds = int(args[i + 1])
                i += 2
            elif args[i] == "--dataset":
                dataset = args[i + 1]
                i += 2
            else:
                ns.append(int(args[i]))
                i += 1

        for n in ns:
            solved = 0
            best_e = 1 << 60
            print(f"\n--- {strategy} n={n} ({num_seeds} seeds) ---")
            for s in range(num_seeds):
                print(f"  seed={s}:", end=" ", flush=True)
                r = bench_n(strategy, strategy, n, 200_000, s, dataset=dataset)
                if r["solved"]:
                    solved += 1
                best_e = min(best_e, r["energy"])
                status = "OK" if r["solved"] else f"e={r['energy']}"
                print(f"{status}  {_fmt_time(r['elapsed'])}")
            print(f"  => {solved}/{num_seeds}, best_e={best_e}")
        return

    # full sweep: 2er Schritte, teure zuerst
    gs4_ns = list(range(32, 7, -1))
    golay_ns = list(range(34, 5, -2))

    configs: list[tuple[str, str, list[int], int]] = [
        ("gs4", "gs4", gs4_ns, 200_000),
        ("golay_2n", "golay_2n", golay_ns, 200_000),
    ]

    results = []
    for strategy, kind, ns, steps in configs:
        for n in ns:
            print(f"Benchmarking {strategy} n={n}...", end=" ", flush=True)
            r = bench_n(strategy, kind, n, steps, seed=42)
            status = "OK" if r["solved"] else f"e={r['energy']}"
            iters = r["iterations"]
            t_fmt = _fmt_time(r["elapsed"])
            print(f"{status}  iters={iters}  {t_fmt}")
            results.append(r)

    # write BENCHMARKS.md
    lines = ["# Benchmarks\n"]
    lines.append("| Date | Commit | Strategy | n | Order | Solved | Energy | Iters | Elapsed |")
    lines.append("|------|--------|----------|--:|-------|--------|--------|------:|--------:|")
    for r in results:
        solved = "YES" if r["solved"] else "NO"
        lines.append(
            f"| {r['date']} | {r['commit']} | {r['strategy']} | {r['n']} | "
            f"{r['order']} | {solved} | {r['energy']} | {r['iterations']} | {_fmt_time(r['elapsed'])} |"
        )

    with open("BENCHMARKS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    solved_count = sum(1 for r in results if r["solved"])
    print(f"\nSolved: {solved_count}/{len(results)}")


if __name__ == "__main__":
    main()
