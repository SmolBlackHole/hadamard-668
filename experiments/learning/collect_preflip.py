"""Collect pre-solve states at n=44 with the instrumented solver.

For every run that solves, record the IMMEDIATE state before the solving
flip: u_pre, d_solve, Q_pre, supp(u_pre) and the verification d == -u.
Appends one JSON line per run to preflip_n44.jsonl (all runs, solved flag
included; only solved rows carry the pre-solve block).

Usage:  python collect_preflip.py <start_seed> <count> [worker_id]
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflip_solver import random_seqs, search_preflip  # noqa: E402
from solver_snapshot.solver import SolverConfig  # noqa: E402

HERE = Path(__file__).resolve().parent


def run(seed: int, n: int = 44) -> dict:
    cfg = SolverConfig(targeted_escape=True)
    rng = np.random.default_rng(seed)
    seqs = random_seqs(n, rng)
    log: dict = {}
    solve_rec: dict = {}
    walk_log: list = []
    t0 = time.perf_counter()
    best_seq, best_e, used, stats = search_preflip(
        seqs, None, rng, steps=100000, config=cfg, log=log, solve_rec=solve_rec,
        walk_log=walk_log,
    )
    dt = time.perf_counter() - t0
    out = {
        "seed": seed, "n": n, "solved": best_e == 0, "best_q": best_e // (64 * n),
        "time_s": round(dt, 2), "q0": log.get("q0"),
        "lowest_q": log.get("lowest_q_seen"),
        "solved_step": log.get("solved_step"),
        "final_b64": base64.b64encode(best_seq.tobytes()).decode(),
    }
    if best_e == 0:
        out["pre"] = solve_rec
    # keep the last walks (incl. the solving one) for deficit-trajectory stats
    out["walks"] = walk_log[-40:]
    return out


def main() -> None:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    worker = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    f = HERE / "preflip_n44.jsonl"
    done = set()
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["seed"])
            except Exception:
                pass
    seeds = [s for s in range(start, start + count) if s not in done]
    print(f"collecting {len(seeds)} preflip runs at n=44 (worker {worker})", flush=True)
    n_solved = 0
    for i, seed in enumerate(seeds):
        r = run(seed)
        n_solved += int(r["solved"])
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n")
        if (i + 1) % 10 == 0:
            print(f"  worker{worker}: {i+1}/{len(seeds)} done, solved so far "
                  f"{n_solved} (last seed {seed}: solved={r['solved']}, "
                  f"{r['time_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
