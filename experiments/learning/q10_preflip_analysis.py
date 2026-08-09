"""Q10: d = -u theorem validation + exception-set analysis + C-distance.

Part 1: validate on real solutions (preflip_n44.jsonl) that the solving flip
satisfies d_solve == -u_pre exactly, and that the pre-state obeys the
mirror-with-exception-set structure:
    u_t = 0  -> x[c-t] = -x[c+t]
    u_t = +1 -> x[c-t] = x[c+t] = x[c]
    u_t = -1 -> x[c-t] = x[c+t] = -x[c]

Part 2: exception sets: |supp(u_pre)| distribution, Q_pre distribution,
which Q-levels solve in one flip, exception-lag positions.

Part 3: cancellation manifold C = {x : exists (s,c), d_{s,c}(x) = -u(x)}.
For the collected escape states (n=44 spec replay + n=52 all_seeds_escape):
  dist0 = state itself in C,
  dist1 = a single flip reaches a state in C,
  min_1flip_Q = min over flips of the resulting Q (defect proxy),
and correlate with solve success.

Usage: python q10_preflip_analysis.py
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

HERE = Path(__file__).resolve().parent


def delta_all(seqs: np.ndarray) -> np.ndarray:
    """Full single-flip delta matrix (4n, m) for a state."""
    n = seqs.shape[1]
    m = (n - 1) // 2
    c = np.arange(n)[:, None]
    t = np.arange(1, m + 1)[None, :]
    forward = (c + t) % n
    backward = (c - t) % n
    fwd_sign = np.where(c + t < n, 1, -1)
    bwd_sign = np.where(backward + t < n, 1, -1)
    v = seqs.astype(np.int16)
    d = -2 * v[:, :, None] * (fwd_sign[None, :, :] * v[:, forward] +
                              bwd_sign[None, :, :] * v[:, backward])
    return (d // 4).reshape(4 * n, m)


def neighbor_deltas(seqs: np.ndarray) -> np.ndarray:
    """Delta matrices of all 4n single-flip neighbors, shape (4n, 4n, m)."""
    n = seqs.shape[1]
    m = (n - 1) // 2
    N = 4 * n
    v = seqs.astype(np.int16)  # (4, n)
    v_all = np.broadcast_to(v, (N, 4, n)).copy()
    s_all = np.repeat(np.arange(4), n)
    c_all = np.tile(np.arange(n), 4)
    v_all[np.arange(N), s_all, c_all] *= -1
    c = np.arange(n)[:, None]
    t = np.arange(1, m + 1)[None, :]
    forward = (c + t) % n
    backward = (c - t) % n
    fwd_sign = np.where(c + t < n, 1, -1)
    bwd_sign = np.where(backward + t < n, 1, -1)
    d = -2 * v_all[:, :, :, None] * (
        fwd_sign[None, None, :, :] * v_all[:, :, forward] +
        bwd_sign[None, None, :, :] * v_all[:, :, backward])
    return (d // 4).reshape(N, N, m)


def dist_to_C(seqs: np.ndarray, u: np.ndarray) -> dict:
    """Distance of a state to the cancellation manifold (exact 0/1, defect)."""
    n = seqs.shape[1]
    m = (n - 1) // 2
    D = delta_all(seqs)  # (4n, m)
    target = -u.astype(np.int16)
    dist0 = int(np.sum(np.all(D == target, axis=1)))
    delta_q = 2 * (D.astype(np.int32) @ u.astype(np.int32)) + (D.astype(np.int32) ** 2).sum(1)
    min_1flip_q = int((np.dot(u, u) + delta_q).min())
    dist1 = 0
    if dist0 == 0:
        DN = neighbor_deltas(seqs)  # (4n, 4n, m)
        up = u.astype(np.int16) + D  # u of each neighbor
        tn = -up
        dist1 = int(np.any(np.all(DN == tn[:, None, :], axis=2), axis=1).sum())
    return {"dist0": dist0, "dist1": dist1, "min_1flip_q": min_1flip_q,
            "defect": int(np.sum((D.astype(np.int32) + u.astype(np.int32)) ** 2, axis=1).min())}


# --- Part 1+2: validate d = -u on real solutions -----------------------------


def analyze_preflip(rows: list[dict]) -> dict:
    sols = [r for r in rows if r["solved"] and r.get("pre")]
    res = {"n_runs": len(rows), "n_solved": sum(r["solved"] for r in rows),
           "n_with_pre": len(sols)}
    paths = {}
    q_pre_hist = {}
    supp_hist = {}
    lag_hist = np.zeros(21, dtype=int)  # n=44: m=21
    d_ok = 0
    mirror_ok = 0
    walk_info = []
    max_umax = 0
    for r in sols:
        p = r["pre"]
        paths[p["path"]] = paths.get(p["path"], 0) + 1
        q_pre_hist[p["q_pre"]] = q_pre_hist.get(p["q_pre"], 0) + 1
        supp_hist[p["supp_size"]] = supp_hist.get(p["supp_size"], 0) + 1
        for t in p["supp"]:
            lag_hist[t] += 1
        d_ok += int(p["d_equals_minus_u"])
        max_umax = max(max_umax, max(abs(a) for a in p["u_pre"]))
        # mirror-with-exception verification on the pre-state
        seqs = np.frombuffer(base64.b64decode(p["pre_b64"]), dtype=np.int8).reshape(4, 44)
        s, c = p["s"], p["c"]
        v = seqs[s].astype(np.int16)
        xc = int(v[c])
        u_pre = np.array(p["u_pre"], dtype=np.int32)
        ok = True
        for t in range(1, 22):
            fp = (c + t) % 44
            bp = (c - t) % 44
            fv = int(v[fp]) * (1 if c + t < 44 else -1)
            bv = int(v[bp]) * (1 if bp + t < 44 else -1)
            ut = int(u_pre[t - 1])
            if ut == 0:
                if fv != -bv:
                    ok = False
            else:
                if not (fv == bv == ut * xc):
                    ok = False
        mirror_ok += int(ok)
        if p["path"] == "tabu":
            walk_info.append((p.get("walk_entry_q"), p.get("flip_in_walk"), p["q_pre"]))
    res["paths"] = paths
    res["q_pre_hist"] = {str(k): v for k, v in sorted(q_pre_hist.items())}
    res["supp_size_hist"] = {str(k): v for k, v in sorted(supp_hist.items())}
    res["d_equals_minus_u_all"] = d_ok == len(sols) if sols else None
    res["mirror_exception_ok_all"] = mirror_ok == len(sols) if sols else None
    res["max_abs_u_pre"] = max_umax
    res["lag_exception_hist"] = lag_hist.tolist()
    if walk_info:
        we = np.array([w[0] for w in walk_info])
        fiw = np.array([w[1] for w in walk_info])
        qp = np.array([w[2] for w in walk_info])
        res["tabu_walk_entry_q_hist"] = {str(k): int(v) for k, v in
                                         sorted({int(x): int((we == x).sum()) for x in set(we)}.items())}
        res["tabu_flip_in_walk_mean"] = float(fiw.mean())
        res["tabu_flip_in_walk_max"] = int(fiw.max())
        res["tabu_entry_vs_pre_q"] = [[int(a), int(b)] for a, b in zip(we, qp)]
        # uphill-excursion analysis: q_pre vs q_min seen in the walk
        qmins = []
        for r in sols:
            p = r["pre"]
            if p["path"] == "tabu" and "q_min_before_solve" in p:
                qmins.append((p["q_pre"], p["q_min_before_solve"],
                              p.get("walk_entry_q"), p.get("flip_in_walk")))
        if qmins:
            qps = np.array([a for a, _, _, _ in qmins])
            qms = np.array([b for _, b, _, _ in qmins])
            qes = np.array([c for _, _, c, _ in qmins])
            res["n_with_qmin"] = len(qmins)
            res["pre_q_above_walk_min"] = int((qps > qms).sum())
            res["pre_q_above_entry"] = int((qps > qes).sum())
            res["pre_q_below_entry"] = int((qps < qes).sum())
            res["qmin_vs_qpre_pairs"] = [[int(a), int(b)] for a, b in zip(qps, qms)]
            # trajectory: mean q at relative position just before solving
            trajs = []
            for r in sols:
                p = r["pre"]
                if p["path"] == "tabu" and "q_traj" in p and p["q_traj"]:
                    trajs.append(np.array(p["q_traj"], dtype=np.int64))
            if trajs:
                L = max(len(t) for t in trajs)
                mat = np.full((len(trajs), L), np.nan)
                for i, t in enumerate(trajs):
                    mat[i, : len(t)] = t
                tail = mat[:, -10:]
                res["traj_tail_mean"] = np.nanmean(tail, axis=0).tolist()
                res["traj_tail_q25"] = np.nanpercentile(tail, 25, axis=0).tolist()
                res["traj_tail_q75"] = np.nanpercentile(tail, 75, axis=0).tolist()
                # median q_pre relative to the trajectory's running min
                res["traj_len_mean"] = float(np.mean([len(t) for t in trajs]))
    return res


# --- Part 3: cancellation manifold distance ----------------------------------


def analyze_c_manifold_n44() -> dict:
    rows = [json.loads(l) for l in (HERE / "collect_n44_spec.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    solved_by_seed = {}
    if (HERE / "collect_n44.jsonl").exists():
        for l in (HERE / "collect_n44.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            solved_by_seed[r["seed"]] = r["solved"]
    stats = {"n_escape": 0, "solved": 0, "unsolved": 0,
             "dist0_solved": 0, "dist0_unsolved": 0,
             "dist1_solved": 0, "dist1_unsolved": 0,
             "min1q_solved": [], "min1q_unsolved": [], "defect_solved": [],
             "defect_unsolved": []}
    dists = []
    for r in rows:
        if "esc_b64" not in r or "esc_u" not in r:
            continue
        seqs = np.frombuffer(base64.b64decode(r["esc_b64"]), dtype=np.int8).reshape(4, 44)
        u = np.array(r["esc_u"], dtype=np.int32)
        d = dist_to_C(seqs, u)
        solved = bool(solved_by_seed.get(r["seed"], False))
        stats["n_escape"] += 1
        stats["solved" if solved else "unsolved"] += 1
        stats["dist0_solved" if solved else "dist0_unsolved"] += int(d["dist0"] > 0)
        stats["dist1_solved" if solved else "dist1_unsolved"] += int(d["dist1"] > 0)
        stats["min1q_solved" if solved else "min1q_unsolved"].append(d["min_1flip_q"])
        stats["defect_solved" if solved else "defect_unsolved"].append(d["defect"])
        dists.append((r["seed"], solved, d))
    for k in ("min1q_solved", "min1q_unsolved", "defect_solved", "defect_unsolved"):
        arr = np.array(stats[k])
        if arr.size:
            stats[k] = {"mean": float(arr.mean()), "min": int(arr.min()),
                        "p25": float(np.percentile(arr, 25)),
                        "median": float(np.median(arr)), "max": int(arr.max())}
        else:
            stats[k] = None
    # correlation: min_1flip_q and defect vs solved (point-biserial)
    y = np.array([d[1] for d in dists], dtype=float)
    for key, field in (("min1q", "min_1flip_q"), ("defect", "defect")):
        x = np.array([d[2][field] for d in dists], dtype=float)
        if len(np.unique(x)) > 1:
            stats[f"corr_{key}_solved"] = float(np.corrcoef(x, y)[0, 1])
    return stats


def analyze_c_manifold_n52() -> dict:
    d = json.load(open(HERE / "all_seeds_escape.json"))
    stats = {"n_escape": 0, "solved": 0, "unsolved": 0, "dist0_solved": 0,
             "dist0_unsolved": 0, "dist1_solved": 0, "dist1_unsolved": 0,
             "min1q_solved": [], "min1q_unsolved": []}
    for k, v in d.items():
        if "seqs_at_escape_b64" not in v:
            continue
        seqs = np.frombuffer(base64.b64decode(v["seqs_at_escape_b64"]), dtype=np.int8).reshape(4, 52)
        u = np.array(json.loads(v["u_at_escape"]) if isinstance(v["u_at_escape"], str) else v["u_at_escape"],
                     dtype=np.int32)
        sol = dist_to_C(seqs, u)
        solved = bool(v["solved"])
        stats["n_escape"] += 1
        stats["solved" if solved else "unsolved"] += 1
        stats["dist0_solved" if solved else "dist0_unsolved"] += int(sol["dist0"] > 0)
        stats["dist1_solved" if solved else "dist1_unsolved"] += int(sol["dist1"] > 0)
        stats["min1q_solved" if solved else "min1q_unsolved"].append(sol["min_1flip_q"])
    for k in ("min1q_solved", "min1q_unsolved"):
        arr = np.array(stats[k])
        stats[k] = ({"mean": float(arr.mean()), "median": float(np.median(arr)),
                     "max": int(arr.max())} if arr.size else None)
    return stats


def main() -> None:
    out: dict = {}
    pf = HERE / "preflip_n44.jsonl"
    if pf.exists():
        rows = [json.loads(l) for l in pf.read_text(encoding="utf-8").splitlines()]
        out["preflip"] = analyze_preflip(rows)
        print(json.dumps(out["preflip"], indent=2, default=str))
    out["c_manifold_n44"] = analyze_c_manifold_n44()
    out["c_manifold_n52"] = analyze_c_manifold_n52()
    print("\n--- cancellation manifold n=44 ---")
    print(json.dumps(out["c_manifold_n44"], indent=2, default=str))
    print("\n--- cancellation manifold n=52 ---")
    print(json.dumps(out["c_manifold_n52"], indent=2, default=str))
    (HERE / "q10_results.json").write_text(json.dumps(out, indent=2, default=str),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
