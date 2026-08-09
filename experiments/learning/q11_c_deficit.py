"""Q11: C-deficit trajectories of tabu walks.

Every tabu walk now records, per step, the C-deficit: the minimum Q
reachable by ONE flip from the state at that step (0 = state is on the
Cancellation Manifold C).  Questions:

A) Within solving walks: does the deficit decrease toward the end
   (walk "approaches C" gradually -> steerable) or stay high until the
   very last step (C hit by chance)?
B) Solving vs non-solving walks: are there observable deficit signatures
   (min deficit, final deficit, deficit at offset k) that distinguish a
   solving walk before it solves?

Usage: python q11_c_deficit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = [json.loads(l) for l in (HERE / "preflip_n44.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    out: dict = {}
    solve_walks: list[list[int]] = []
    nonsolve_walks: list[list[int]] = []
    n_runs = 0
    for r in rows:
        if "walks" not in r:
            continue
        n_runs += 1
        for w in r["walks"]:
            d = np.array(w.get("deficit", []), dtype=np.int64)
            if d.size < 2:
                continue
            if w.get("solved"):
                solve_walks.append(d)
            else:
                nonsolve_walks.append(d)

    out["n_runs_with_walklog"] = n_runs
    out["n_solving_walks"] = len(solve_walks)
    out["n_nonsolving_walks"] = len(nonsolve_walks)

    def describe(arr: np.ndarray) -> dict:
        return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "min": int(arr.min()), "max": int(arr.max()),
                "n": int(arr.size)}

    # A) within-walk: deficit at offsets before the end
    tails = {}
    for off in (1, 2, 3, 4, 5, 8, 12, 20):
        vals = []
        for d in solve_walks:
            if d.size > off:
                vals.append(d[-off - 1])
        if vals:
            tails[f"deficit_{off}_steps_before_solve"] = describe(np.array(vals))
    out["solving_walk_tails"] = tails

    # min deficit over the walk EXCLUDING the final 0 (the solving step)
    min_before = []
    for d in solve_walks:
        pre = d[:-1]
        if pre.size:
            min_before.append(pre.min())
    out["solving_walk_min_deficit_before_solve"] = describe(np.array(min_before))

    # how many solving walks reach deficit 1 or 2 before the final step?
    hit1 = sum(1 for d in solve_walks if (d[:-1] == 1).any())
    hit2 = sum(1 for d in solve_walks if ((d[:-1] <= 2) & (d[:-1] >= 0)).any())
    out["solve_walks_reaching_deficit_1"] = hit1
    out["solve_walks_reaching_deficit_<=2"] = hit2

    # B) comparison: non-solving walks final deficit & min deficit
    nsf = np.array([d[-1] for d in nonsolve_walks])
    nsm = np.array([d.min() for d in nonsolve_walks])
    out["nonsolving_walk_final_deficit"] = describe(nsf)
    out["nonsolving_walk_min_deficit"] = describe(nsm)
    sf = np.array([d[-1] for d in solve_walks])
    out["solving_walk_final_deficit"] = describe(sf)

    # within-walk trend: Spearman-style correlation of step vs deficit
    trends = []
    for d in solve_walks:
        if d.size > 10:
            x = np.arange(d.size)
            trends.append(float(np.corrcoef(x, d)[0, 1]))
    out["solving_walk_step_deficit_corr_mean"] = float(np.mean(trends)) if trends else None

    # last-walk-of-unsolved-run: how close does a failed run get to C?
    last_def = []
    for r in rows:
        if r.get("solved") or "walks" not in r:
            continue
        ws = [w for w in r["walks"] if not w.get("solved")]
        if ws:
            last_def.append(ws[-1]["deficit"][-1])
    out["unsolved_run_last_walk_final_deficit"] = describe(np.array(last_def)) \
        if last_def else None

    # C-density along tabu trajectories: fraction of walk steps on C
    def c_density(walks):
        steps = 0
        on_c = 0
        for d in walks:
            steps += d.size
            on_c += int((d == 0).sum())
        return (float(on_c / steps) if steps else None, int(on_c), int(steps))

    out["C_density_solving_walks"] = c_density(solve_walks)
    out["C_density_nonsolving_walks"] = c_density(nonsolve_walks)
    # density of deficit == 1 (one flip away from C, "almost on C")
    def near_c_density(walks):
        steps = 0
        near = 0
        for d in walks:
            steps += d.size
            near += int((d == 1).sum())
        return (float(near / steps) if steps else None, int(near), int(steps))

    out["deficit1_density_solving_walks"] = near_c_density(solve_walks)
    out["deficit1_density_nonsolving_walks"] = near_c_density(nonsolve_walks)

    print(json.dumps(out, indent=1, default=str))
    (HERE / "q11_results.json").write_text(json.dumps(out, indent=1, default=str),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
