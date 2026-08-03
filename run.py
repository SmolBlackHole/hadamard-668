"""Hadamard-668 search — einziger Entrypoint.

    python run.py                           # default: williamson, 200k steps, 1 job
    python run.py --jobs 8                  # 8 parallele Suchen
    python run.py --strategy direct         # direkte Suche
    python run.py --steps 50000 --seed 99   # kurzer Testlauf
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

# src als paketpfad
sys.path.insert(0, str(Path(__file__).parent / "src"))


def _worker_single(args: tuple[int, int, str, str]) -> dict:
    seed, steps, output_dir, strategy = args
    os.environ["BATCH"] = "1"  # unterdrueckt tqdm in subprozessen
    t0 = time.perf_counter()
    from search import direct_search, williamson_search, ORDER

    if strategy == "direct":
        matrix, metrics, _elapsed = direct_search(steps=steps, seed=seed)
    else:
        matrix, metrics, _elapsed = williamson_search(steps=steps, seed=seed)

    out = Path(output_dir) / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    lines = [",".join(str(int(v)) for v in row) for row in matrix]
    (out / "candidate.csv").write_text("\n".join(lines) + "\n")

    wall = round(time.perf_counter() - t0, 1)
    result = {
        "seed": seed,
        "steps": steps,
        "strategy": strategy,
        "wall_seconds": wall,
        "energy": metrics["energy"],
        "orthogonal_pairs": metrics["orthogonal_pairs"],
        "max_abs_correlation": metrics["max_abs_correlation"],
        "total_pairs": ORDER * (ORDER - 1) // 2,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description="Hadamard-668 search")
    p.add_argument("--strategy", choices=["direct", "williamson", "both"], default="williamson")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=1, help="parallele Prozesse (>1 = batch)")
    p.add_argument("--runs-dir", type=str, default="runs")
    args = p.parse_args()

    jobs = max(1, args.jobs)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if jobs == 1:
        print(f"Single run  strategy={args.strategy}  steps={args.steps}  seed={args.seed}\n")
        results = [_worker_single((args.seed, args.steps, args.runs_dir, args.strategy))]
    else:
        seeds = list(range(args.seed, args.seed + jobs * 4))
        print(f"Batch  jobs={jobs}  seeds={len(seeds)}  steps={args.steps}\n")

        tasks = [(s, args.steps, args.runs_dir, args.strategy) for s in seeds]
        t0 = time.perf_counter()
        with multiprocessing.Pool(processes=jobs) as pool:
            results = list(pool.imap_unordered(_worker_single, tasks))
        print(f"\nBatch done: {len(results)} runs in {time.perf_counter() - t0:.0f}s")

    results.sort(key=lambda r: r["energy"])
    best = results[0]

    print(f"\n--- Results ---")
    for r in results[:15]:
        m = " <--" if r is best else ""
        print(f"  seed={r['seed']:>4}  energy={r['energy']:>10}  orth={r['orthogonal_pairs']:>7}  max={r['max_abs_correlation']:>4}  {r['wall_seconds']:>6.1f}s{m}")
    if len(results) > 15:
        print(f"  ... +{len(results) - 15} more")

    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    best_file = Path(args.runs_dir) / f"best_{ts}.json"
    best_file.write_text(json.dumps(best, indent=2) + "\n")
    print(f"\nBest: seed={best['seed']}  energy={best['energy']}")
    if best["energy"] == 0:
        print("*** HADAMARD! ***")


if __name__ == "__main__":
    main()
