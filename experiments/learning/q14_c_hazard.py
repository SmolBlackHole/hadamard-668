"""Q14: C-hit hazard analysis.

Is P(C-hit | Q=q, Q-Q_min=h, steps_since_kick=t) flat or does it peak in
certain Q-shells?  Two views:

A) CONDITIONAL per-step hazard from the solving walks (preflip files):
   each solving walk's q_traj gives the Q of the state at every walk
   position; the last position is the C-hit.  h(q) = hits_at(q)/steps_at(q)
   is the per-step hit rate among states with Q=q (length-biased, but it
   shows WHERE on the walk the hits concentrate).
   Additionally the 2D table (Q_pre, Q_min) and the steps-to-hit
   distribution (flip_in_walk) -> "steps_since_kick" lever.

B) UNCONDITIONAL walk-level hazard from ALL logged walks:
   P(hit in walk | entry Q band).  For preflip_n44.jsonl the log is the
   last-40-walks sample (75 runs, 47 solving / 2953 non-solving) — biased
   toward the final regime but the regime where hits actually occur.
   Runs collected with collect_preflip_n.py carry walk_count = the FULL
   per-run entry-Q histogram (n_hits, n_walks per entry Q) — unbiased.

Runs on preflip_n44.jsonl and, if present, preflip_n50.jsonl.

Usage: python q14_c_hazard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def analyze_file(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    sols = [r for r in rows if r.get("solved") and r.get("pre")]
    n = rows[0]["n"] if rows else 0
    out: dict = {"file": path.name, "n_runs": len(rows),
                 "n_solved": sum(r.get("solved", False) for r in rows)}

    # --- per-walk state-Q reconstruction from solving walks -------------------
    # state at walk position p has Q: entry_q (p=0), else q_traj[p-1].
    # hit at position L-1 (L = flip_in_walk).
    steps_at: dict[int, int] = {}
    hits_at: dict[int, int] = {}
    hit_steps: list[int] = []
    qmin_pairs: list[tuple[int, int]] = []
    t_at_risk: list[int] = []
    t_hits: list[int] = []
    for r in sols:
        p = r["pre"]
        if p["path"] != "tabu":
            continue
        L = int(p.get("flip_in_walk") or 0)
        entry = int(p.get("walk_entry_q", -1))
        q_pre = int(p["q_pre"])
        q_min = int(p.get("q_min_before_solve", -1))
        traj = [int(v) for v in p.get("q_traj", [])]
        qmin_pairs.append((q_min, q_pre))
        hit_steps.append(L)
        # The hit Q is logged exactly (q_pre); the per-step hazard needs the
        # full trajectory, which only batches 2+3 carry.  Walks without
        # q_traj contribute to hit_steps/qmin but not to the per-step counts.
        if not traj or len(traj) < L:
            continue
        # state at walk position k has Q: entry (k=0) else traj[k-1].
        # traj[-1] == 0 (the solving flip); hit state is position L-1.
        if L == 1:
            q_at_hit = entry
        else:
            q_at_hit = traj[L - 2]
        hits_at[q_at_hit] = hits_at.get(q_at_hit, 0) + 1
        for k in range(L):
            qk = entry if k == 0 else traj[k - 1]
            steps_at[qk] = steps_at.get(qk, 0) + 1
            if k < L - 1:
                t_at_risk.append(k + 1)
        t_hits.append(L)
    out["n_solving_walks"] = len(hit_steps)
    out["n_solving_walks_with_traj"] = len(t_hits)

    # conditional per-step hazard h(q) = hits_at(q) / steps_at(q)
    qs = sorted(set(steps_at) | set(hits_at))
    haz = {}
    for q in qs:
        st = steps_at.get(q, 0)
        hi = hits_at.get(q, 0)
        haz[str(q)] = {"steps_at": st, "hits": hi,
                       "h_per_step": (hi / st) if st else None}
    out["conditional_hazard_by_q"] = haz
    tot_steps = sum(steps_at.values())
    tot_hits = sum(hits_at.values())
    out["hazard_steps_total"] = tot_steps
    out["hazard_hits_total"] = tot_hits
    if tot_steps:
        out["overall_h_per_step_cond"] = tot_hits / tot_steps

    # steps-to-hit (steps_since_kick): conditional distribution
    hs = np.array(hit_steps, dtype=float)
    out["steps_to_hit"] = {"mean": float(hs.mean()), "median": float(np.median(hs)),
                           "p25": float(np.percentile(hs, 25)),
                           "p75": float(np.percentile(hs, 75)),
                           "max": int(hs.max()) if hs.size else None,
                           "hist": {str(k): int((hs == k).sum())
                                    for k in range(int(hs.min()), int(hs.max()) + 1)}
                           if hs.size else {}}
    # time-varying hazard over walk position (conditional): h(t)=hits_t/at_risk_t
    if t_at_risk:
        ar = np.bincount(t_at_risk)
        th = np.bincount(t_hits, minlength=ar.size)
        band = {}
        for lo in (1, 21, 41, 61, 81, 101, 121, 141, 161, 181):
            hi = lo + 19
            a = int(ar[lo - 1 : hi].sum()) if hi <= ar.size else int(ar[lo - 1 :].sum())
            h = int(th[lo - 1 : hi].sum()) if hi <= ar.size else int(th[lo - 1 :].sum())
            band[f"{lo}-{min(hi, ar.size)}"] = {"at_risk": a, "hits": h,
                                                "h": h / a if a else None}
        out["positional_hazard_bands"] = band

    # --- (Q_pre, Q-Q_min) table: the hit geometry -----------------------------
    qm = np.array([a for a, _ in qmin_pairs], dtype=int)
    qp = np.array([b for _, b in qmin_pairs], dtype=int)
    out["n_with_qmin"] = len(qmin_pairs)
    if len(qmin_pairs):
        exc = qp - qm  # uphill excursion size at the hit
        out["hit_excursion_h"] = {"mean": float(exc.mean()),
                                  "median": float(np.median(exc)),
                                  "hist": {str(k): int((exc == k).sum())
                                           for k in range(int(exc.min()), int(exc.max()) + 1)}}
        out["hit_above_walk_min"] = int((exc > 0).sum())
        # 2D table: rows = Q_min band, cols = Q_pre
        tbl = {}
        for a, b in qmin_pairs:
            row = tbl.setdefault(str(int(a)), {})
            row[str(int(b))] = row.get(str(int(b)), 0) + 1
        out["qmin_x_qpre"] = tbl

    # --- B) unconditional walk-level hazard from the logged walks -------------
    n_walks = 0
    n_hits = 0
    by_entry: dict[str, list[int]] = {}
    for r in rows:
        for w in r.get("walks", []):
            eq = w.get("entry_q")
            if eq is None:
                continue
            n_walks += 1
            hit = int(bool(w.get("solved")))
            n_hits += hit
            b = by_entry.setdefault(str(int(eq)), [0, 0])
            b[0] += 1
            b[1] += hit
    out["logged_walks"] = n_walks
    out["logged_walk_hits"] = n_hits
    if n_walks:
        out["logged_walk_hit_rate"] = n_hits / n_walks
    out["walk_hazard_by_entry_q"] = {k: {"n": v[0], "hits": v[1],
                                         "p_hit": v[1] / v[0] if v[0] else None}
                                     for k, v in sorted(by_entry.items(),
                                                        key=lambda kv: int(kv[0]))}

    # --- full-run walk counts (unbiased) if present ---------------------------
    full: dict[str, list[int]] = {}
    n_runs_full = 0
    for r in rows:
        wc = r.get("walk_count")
        if not wc:
            continue
        n_runs_full += 1
        for k, v in wc.items():
            b = full.setdefault(k, [0, 0])
            b[0] += v[0]
            b[1] += v[1]
    if full:
        out["n_runs_with_full_walk_counts"] = n_runs_full
        out["full_walk_hazard_by_entry_q"] = {k: {"n": v[0], "hits": v[1],
                                                  "p_hit": v[1] / v[0] if v[0] else None}
                                              for k, v in sorted(full.items(),
                                                                 key=lambda kv: int(kv[0]))}
        tot = sum(v[0] for v in full.values())
        htot = sum(v[1] for v in full.values())
        out["full_walks_total"] = tot
        out["full_walk_hit_rate"] = htot / tot if tot else None
        # share of walks at entry Q <= 13 (the "low-Q regime")
        low = sum(v[0] for k, v in full.items() if int(k) <= 13)
        out["share_walks_entry_le13"] = low / tot if tot else None
        lowh = sum(v[1] for k, v in full.items() if int(k) <= 13)
        out["share_hits_entry_le13"] = lowh / htot if htot else None
        # solve rate vs total walks per run
        wpr = [sum(v[0] for v in r["walk_count"].values())
               for r in rows if r.get("walk_count")]
        out["walks_per_run"] = {"mean": float(np.mean(wpr)) if wpr else None,
                                "median": float(np.median(wpr)) if wpr else None,
                                "max": int(max(wpr)) if wpr else None}
    return out


def main() -> None:
    out: dict = {}
    for name in ("preflip_n44.jsonl", "preflip_n50.jsonl"):
        p = HERE / name
        if p.exists() and p.stat().st_size:
            print(f"analyzing {name} ...", flush=True)
            out[name] = analyze_file(p)
    print(json.dumps(out, indent=1, default=str))
    (HERE / "q14_results.json").write_text(json.dumps(out, indent=1, default=str),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
