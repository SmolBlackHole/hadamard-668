"""Q12: Can C be reached constructively?  Steepest-descent test.

The C-deficit analysis (q11) shows the tabu walk does not approach C
gradually.  Construction question: does always taking the single flip that
MINIMIZES the resulting Q (steepest descent, global min over all 4n flips,
improving moves only) reach a solution (Q=0) from the collected escape
states?  This is the strongest "local steering" candidate for a
construction rule.  Baseline for comparison: the tabu walk solves ~60% of
runs.

We also test a "steepest-descent with restarts" variant: after a local
minimum, apply a random kick of 4 flips (like the solver's outer loop)
and continue, for a comparable step budget.

Usage: python q12_steepest_descent.py
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solver_snapshot.tracker import Tracker  # noqa: E402

HERE = Path(__file__).resolve().parent


def steepest_descent(seqs: np.ndarray, max_steps: int = 200) -> tuple[int, int]:
    """Always flip the global Q-minimizing single flip (improving only)."""
    tracker = Tracker()
    tracker.build(seqs)
    cur = seqs.copy()
    q = tracker._q
    q_min = q
    steps = 0
    for _ in range(max_steps):
        qs = tracker.flip_qs()
        best = int(qs.min())
        if best >= q:
            break
        idx = int(np.argmin(qs))
        s, c = divmod(idx, cur.shape[1])
        tracker.accept(cur, s, c)
        q = tracker._q
        q_min = min(q_min, q)
        steps += 1
        if q == 0:
            break
    return q_min, steps


def steepest_with_kicks(seqs: np.ndarray, rng: np.random.Generator,
                        max_steps: int = 2000) -> int:
    """Steepest descent + 4-flip random kick at local minima."""
    tracker = Tracker()
    tracker.build(seqs)
    cur = seqs.copy()
    q = tracker._q
    q_min = q
    n = cur.shape[1]
    steps = 0
    while steps < max_steps and q > 0:
        qs = tracker.flip_qs()
        best = int(qs.min())
        if best >= q:
            # kick: random 4-flip (one per sequence)
            cols = rng.integers(0, n, size=4)
            q = tracker.accept(cur, 0, int(cols[0]))
            q = tracker.accept(cur, 1, int(cols[1]))
            q = tracker.accept(cur, 2, int(cols[2]))
            q = tracker.accept(cur, 3, int(cols[3]))
            q = tracker._q
            steps += 1
        else:
            idx = int(np.argmin(qs))
            s, c = divmod(idx, n)
            q = tracker.accept(cur, s, c)
            steps += 1
        q_min = min(q_min, q)
    return q_min


def main() -> None:
    rows = [json.loads(l) for l in (HERE / "collect_n44_spec.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    solved_by_seed = {}
    if (HERE / "collect_n44.jsonl").exists():
        for l in (HERE / "collect_n44.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            solved_by_seed[r["seed"]] = r["solved"]
    states = [r for r in rows if "esc_b64" in r]
    print(f"testing steepest descent on {len(states)} escape states (n=44)")
    sd_qmin = []
    sd_solved = 0
    sk_qmin = []
    sk_solved = 0
    rng = np.random.default_rng(12345)
    for i, r in enumerate(states):
        seqs = np.frombuffer(base64.b64decode(r["esc_b64"]), dtype=np.int8).reshape(4, 44)
        q1, _ = steepest_descent(seqs)
        q2 = steepest_with_kicks(seqs, max_steps=2000, rng=rng)
        sd_qmin.append(q1)
        sk_qmin.append(q2)
        sd_solved += int(q1 == 0)
        sk_solved += int(q2 == 0)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(states)}", flush=True)
    out = {
        "n_states": len(states),
        "steepest_descent_solved": sd_solved,
        "steepest_descent_minq_mean": float(np.mean(sd_qmin)),
        "steepest_descent_minq_median": float(np.median(sd_qmin)),
        "steepest_descent_minq_hist": {str(k): int(v) for k, v in
                                       sorted({x: sd_qmin.count(x) for x in set(sd_qmin)}.items())},
        "steepest_with_kicks_solved": sk_solved,
        "steepest_with_kicks_minq_mean": float(np.mean(sk_qmin)),
        "steepest_with_kicks_minq_median": float(np.median(sk_qmin)),
    }
    print(json.dumps(out, indent=1))
    (HERE / "q12_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
