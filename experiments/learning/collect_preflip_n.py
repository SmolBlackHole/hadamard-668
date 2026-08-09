"""Collect pre-solve states + FULL tabu-walk entry-Q counts at any n.

Same pre-flip instrumentation as collect_preflip.py, plus:
- out["walk_count"] = {entry_q: [n_walks, n_hits]} over ALL tabu walks of the
  run (the solver appends every walk; collect_preflip only kept the last 40).
  This gives the unbiased walk-level hazard P(C-hit | walk entry Q) and the
  number of walks per run in the low-Q regime.
- out["walks"] keeps the last 40 walks with deficit trajectories (as before).

Usage:  python collect_preflip_n.py <n> <start_seed> <count> [worker_id]
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


class WalkLog:
    """Bounded walk log: aggregates ALL walks into an entry-Q counter but
    keeps only the last `keep` walk dicts (with deficit trajectories).

    The solver can emit ~10k-100k walks per run; keeping every deficit list
    (200 ints each) blew up memory and killed the collectors on pathological
    runs.  The full per-entry-Q counts are what the hazard analysis needs.
    """

    def __init__(self, keep: int = 41) -> None:
        self.keep = keep
        self.buf: list[dict] = []
        self.count: dict[str, list[int]] = {}

    def append(self, w: dict) -> None:
        eq = w.get("entry_q")
        if eq is not None:
            b = self.count.setdefault(str(int(eq)), [0, 0])
            b[0] += 1
            b[1] += int(bool(w.get("solved")))
        self.buf.append(w)
        if len(self.buf) > self.keep:
            del self.buf[: len(self.buf) - self.keep]

    def last(self, n: int) -> list[dict]:
        return self.buf[-n:]

    def __len__(self) -> int:
        return len(self.buf)


def run(seed: int, n: int = 44) -> dict:
    cfg = SolverConfig(targeted_escape=True)
    rng = np.random.default_rng(seed)
    seqs = random_seqs(n, rng)
    log: dict = {}
    solve_rec: dict = {}
    walk_log = WalkLog()
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
        "n_walks_total": sum(v[0] for v in walk_log.count.values()),
        "walk_count": walk_log.count,
        "final_b64": base64.b64encode(best_seq.tobytes()).decode(),
    }
    if best_e == 0:
        out["pre"] = solve_rec
    out["walks"] = walk_log.last(40)
    return out


def main() -> None:
    n = int(sys.argv[1])
    start = int(sys.argv[2])
    count = int(sys.argv[3])
    worker = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    f = HERE / f"preflip_n{n}.jsonl"
    done = set()
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["seed"])
            except Exception:
                pass
    seeds = [s for s in range(start, start + count) if s not in done]
    print(f"collecting {len(seeds)} runs at n={n} (worker {worker})", flush=True)
    n_solved = 0
    for i, seed in enumerate(seeds):
        r = run(seed, n)
        n_solved += int(r["solved"])
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n")
        if (i + 1) % 5 == 0:
            print(f"  worker{worker}: {i+1}/{len(seeds)} done, solved so far "
                  f"{n_solved} (seed {seed}: solved={r['solved']}, "
                  f"{r['time_s']:.1f}s, walks={r['n_walks_total']})", flush=True)


if __name__ == "__main__":
    main()
