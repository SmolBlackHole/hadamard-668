"""Exhaustive correctness and sensitivity tests for syndrome_repair.

Usage::

    python -m experiments.repair_tests [1..6] [--tt 8,16,36] [--quick] [--seed 2026]

Tests:
  1  Model correctness: GPU deltas and Q[i,j] vs exact CPU NPAF
  2  Single-bit sensitivity: flip every position, record energy + repair
  3  Pair sensitivity: all 2-bit perturbations, categorize by seq pair
  4  Repair radius: random errors at exact Hamming distances
  5  Structured errors: one_seq, bunched, edges, spread
  6  Pair utility: singles to stall, measure pair escape rate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

# Bootstrap: add repo root to path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from correlations import (  # noqa: E402
    TURYN_WEIGHTS_I,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
)
from experiments.syndrome_repair import RepairExperiment  # noqa: E402
from fixtures import (  # noqa: E402
    hamming_distance,
    tt_lengths,
    tt_sequences,
)
from gpu import xp  # noqa: E402

SEQ_NAMES = ["A", "B", "C", "D"]
RESULTS_DIR = _REPO / "results"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _exact_residual(seq: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Exact int64 NPAF residual at shifts 1..n-1."""
    return nonperiodic_autocorrelation_state(seq, lengths=lengths, weights=TURYN_WEIGHTS_I)[1:]


def _exact_energy(seq: np.ndarray, lengths: np.ndarray) -> float:
    return float(nonperiodic_correlation_energy(
        nonperiodic_autocorrelation_state(seq, lengths=lengths, weights=TURYN_WEIGHTS_I)))


def _all_positions(lengths: np.ndarray) -> list[tuple[int, int]]:
    return [(r, c) for r in range(4) for c in range(int(lengths[r]))]


def _damage_random(rng: np.random.Generator, sol: np.ndarray,
                   positions: list[tuple[int, int]], d: int) -> np.ndarray:
    """Return ``sol`` with exactly ``d`` randomly chosen bits flipped."""
    damaged = sol.copy()
    chosen = rng.choice(len(positions), size=d, replace=False)
    for idx in chosen:
        r, c = positions[int(idx)]
        damaged[r, c] *= -1
    return damaged


def _emit(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ── Test 1: Model correctness ────────────────────────────────────────────────


def test1_correctness(n_values: list[int], out_dir: Path) -> None:
    """Verify GPU singles/pairs model against exact CPU NPAF."""
    print("\n" + "=" * 60)
    print("TEST 1: Model correctness (GPU vs exact CPU NPAF)")
    print("=" * 60)

    for n in n_values:
        sol = tt_sequences(n)
        lengths = tt_lengths(n)
        B = int(lengths.sum())
        positions = _all_positions(lengths)
        seq_f = xp.asarray(sol.astype(np.float32), dtype=xp.float32)

        rp = RepairExperiment(n=n, sieve=False, verbose=False)
        deltas, r0 = rp._single_deltas(seq_f, positions)

        # 1a: Singles — per-position GPU vs exact
        print(f"\n  TT({n}) — {B} singles positions")
        max_err = 0.0
        max_energy_diff = 0.0
        n_rank_mismatch = 0
        gpu_energies = np.sum((r0 + deltas) ** 2, axis=1)
        exact_energies = np.empty(B, dtype=np.float64)

        for idx, (r, c) in enumerate(positions):
            flipped = sol.copy()
            flipped[r, c] *= -1
            exact_res = _exact_residual(flipped, lengths).astype(np.float64)
            gpu_res = r0 + deltas[idx]
            err = float(np.max(np.abs(gpu_res - exact_res)))
            if err > max_err:
                max_err = err
            exact_energies[idx] = float(np.sum(exact_res ** 2))
            if abs(gpu_energies[idx] - exact_energies[idx]) > max_energy_diff:
                max_energy_diff = abs(gpu_energies[idx] - exact_energies[idx])
            gpu_rank = np.sum(gpu_energies <= gpu_energies[idx])
            exact_rank = np.sum(exact_energies <= exact_energies[idx])
            if gpu_rank != exact_rank:
                n_rank_mismatch += 1

        print(f"    max lag error: {max_err:.6f}")
        print(f"    max energy diff: {max_energy_diff:.2f}")
        print(f"    rank mismatches: {n_rank_mismatch}/{B}")

        if n_rank_mismatch > 0:
            print(
                f"    rank ties (float32 vs exact): {n_rank_mismatch}/{B} (expected at solution states)")

        # 1b: Pairs — GPU Q model vs exact double-flip residual
        print(f"  TT({n}) — pairs")
        if n <= 16:
            # Full enumeration
            n_pairs = B * (B - 1) // 2
            print(f"    full enumeration: {n_pairs} pairs")
            # Build Q for all positions at once
            top_idx = list(range(B))
            Q = rp._q_pair_model(seq_f, positions, top_idx, deltas[:B])
            max_q_err = 0.0
            pair_idx = 0
            for i in range(B):
                for j in range(i + 1, B):
                    # Exact: flip both bits, compute residual
                    dp = sol.copy()
                    ri, ci = positions[i]
                    rj, cj = positions[j]
                    dp[ri, ci] *= -1
                    dp[rj, cj] *= -1
                    exact_res = _exact_residual(dp, lengths).astype(np.float64)
                    gpu_res = r0 + deltas[i] + deltas[j] + Q[i, j]
                    err = float(np.max(np.abs(gpu_res - exact_res)))
                    if err > max_q_err:
                        max_q_err = err
                    pair_idx += 1
                    if pair_idx % 500 == 0:
                        pass  # progress tick
            print(f"    max Q lag error: {max_q_err:.6f}")
        else:
            # Sampled: 3 batches of K=25 random positions
            rng = np.random.default_rng(1)
            for k_size in [25]:
                sampled = rng.choice(B, size=k_size, replace=False).tolist()
                Q = rp._q_pair_model(
                    seq_f, positions, sampled, deltas[sampled])
                max_q_err = 0.0
                for idx_i, i in enumerate(sampled):
                    for idx_j, j in enumerate(sampled[idx_i + 1:], start=idx_i + 1):
                        dp = sol.copy()
                        ri, ci = positions[i]
                        rj, cj = positions[j]
                        dp[ri, ci] *= -1
                        dp[rj, cj] *= -1
                        exact_res = _exact_residual(
                            dp, lengths).astype(np.float64)
                        gpu_res = r0 + deltas[i] + deltas[j] + Q[idx_i, idx_j]
                        err = float(np.max(np.abs(gpu_res - exact_res)))
                        if err > max_q_err:
                            max_q_err = err
            print(f"    max Q lag error (sampled): {max_q_err:.6f}")

    print("\n  => Test 1 complete.")


# ── Test 2: Single-bit sensitivity ────────────────────────────────────────────


def test2_sensitivity_singles(n_values: list[int], out_dir: Path) -> None:
    """Flip every bit individually, record energy + repair outcome."""
    print("\n" + "=" * 60)
    print("TEST 2: Single-bit sensitivity")
    print("=" * 60)

    for n in n_values:
        sol = tt_sequences(n)
        lengths = tt_lengths(n)
        positions = _all_positions(lengths)
        B = len(positions)
        out = out_dir / f"repair_test2_TT{n}.jsonl"

        print(f"\n  TT({n}) — {B} positions")
        for idx, (r, c) in enumerate(positions):
            damaged = sol.copy()
            damaged[r, c] *= -1

            e0 = _exact_energy(damaged, lengths)
            corr = _exact_residual(damaged, lengths)
            violated = int(np.count_nonzero(corr))
            max_res = int(np.abs(corr).max())

            # Repair with singles-only (should always recover)
            rp = RepairExperiment(n=n, sieve=False, pair_top=0, triple_top=0,
                                  plateau_threshold=10 ** 9, verbose=False)
            t0 = time.perf_counter()
            result = rp.search(steps=60, seed=idx, sequences=damaged)
            wall = time.perf_counter() - t0
            success = result.metrics["energy"] == 0
            dist_to_edge = min(c, int(lengths[r]) - 1 - c)
            weight = int(TURYN_WEIGHTS_I[r])

            row = {
                "test": 2, "n": n, "seq": SEQ_NAMES[r], "pos": int(c),
                "dist_to_edge": dist_to_edge, "weight": weight,
                "energy_after_flip": e0, "violated_lags": violated,
                "max_residual": max_res, "repair_success": success,
                "repair_moves": 1 if success else 0,
                "wall_s": wall,
            }
            # Determine exact recovery by direct hamming
            row["hamming_to_original"] = hamming_distance(
                result.sequences, sol, lengths)
            row["exact_recovery"] = row["hamming_to_original"] == 0
            _emit(out, row)

        print(f"    => {out}")


# ── Test 3: Pair sensitivity ──────────────────────────────────────────────────


def test3_sensitivity_pairs(n_values: list[int], quick: bool, out_dir: Path) -> None:
    """All 2-bit perturbations. Categorize, test repair with various pair_top."""
    print("\n" + "=" * 60)
    print("TEST 3: Pair sensitivity")
    print("=" * 60)

    for n in n_values:
        sol = tt_sequences(n)
        lengths = tt_lengths(n)
        positions = _all_positions(lengths)
        B = len(positions)
        out = out_dir / f"repair_test3_TT{n}.jsonl"

        # Choose pair enumeration strategy
        if n <= 8 or n <= 16:
            all_pairs = list(combinations(range(B), 2))
        else:
            # Sample for TT(36): 1200 random pairs
            rng = np.random.default_rng(3)
            all_pairs = [tuple(sorted(rng.choice(B, size=2, replace=False)))
                         for _ in range(1200)]

        print(f"  TT({n}) — {len(all_pairs)} pairs")
        if quick and n > 8:
            all_pairs = all_pairs[:400]
            print(f"    (quick mode: {len(all_pairs)} pairs)")

        for pi, (i, j) in enumerate(all_pairs):
            ri, ci = positions[i]
            rj, cj = positions[j]
            damaged = sol.copy()
            damaged[ri, ci] *= -1
            damaged[rj, cj] *= -1

            e0 = _exact_energy(damaged, lengths)
            cat = SEQ_NAMES[ri] + "+" + SEQ_NAMES[rj]
            same_seq = ri == rj
            dist = abs(ci - cj) if same_seq else -1

            # Test: singles-only vs pairs-full vs pairs-top-64
            row = {"test": 3, "n": n, "i": i, "j": j,
                   "i_seq": SEQ_NAMES[ri], "i_pos": int(ci),
                   "j_seq": SEQ_NAMES[rj], "j_pos": int(cj),
                   "same_seq": same_seq, "distance": dist,
                   "energy_after_flip": e0, "category": cat}

            for cfg_name, cfg_kw in [
                ("singles", {"pair_top": 0, "triple_top": 0,
                 "plateau_threshold": 10 ** 9}),
                ("pairs_full", {"pair_top": B, "triple_top": 0,
                 "plateau_threshold": 10 ** 9}),
                ("pairs_top64", {"pair_top": 64,
                 "triple_top": 0, "plateau_threshold": 10 ** 9}),
            ]:
                rp = RepairExperiment(
                    n=n, sieve=False, verbose=False, **cfg_kw)
                t0 = time.perf_counter()
                result = rp.search(steps=min(200, 30 * B),
                                   seed=pi, sequences=damaged)
                wall = time.perf_counter() - t0
                row[f"{cfg_name}_success"] = result.metrics["energy"] == 0
                row[f"{cfg_name}_energy"] = result.metrics["energy"]
                row[f"{cfg_name}_wall"] = wall
                row[f"{cfg_name}_hamming"] = hamming_distance(
                    result.sequences, sol, lengths)

            _emit(out, row)
            if (pi + 1) % 200 == 0:
                print(f"    {pi + 1}/{len(all_pairs)} done")

        print(f"    => {out}")


# ── Test 4: Repair radius ─────────────────────────────────────────────────────


def test4_repair_radius(n_values: list[int], quick: bool, seed: int, out_dir: Path) -> None:
    """Random errors at exact Hamming distances. Compare pair_top levels."""
    print("\n" + "=" * 60)
    print("TEST 4: Repair radius")
    print("=" * 60)

    d_values = [1, 2, 3, 4, 5, 7, 10, 14, 20, 28]
    n_trials = 3 if quick else 10

    for n in n_values:
        sol = tt_sequences(n)
        lengths = tt_lengths(n)
        positions = _all_positions(lengths)
        B = len(positions)
        out = out_dir / f"repair_test4_TT{n}.jsonl"
        rng = np.random.default_rng(seed)
        print(
            f"  TT({n}) — {B} bits, {n_trials} trials x {len(d_values)} distances")

        for d in d_values:
            if d > B:
                continue
            step_budget = max(120, 30 * d)
            for trial in range(n_trials):
                # Each trial: same damaged state for all configs
                trial_seed = seed + d * n_trials + trial
                damaged = _damage_random(rng, sol, positions, d)
                e0 = _exact_energy(damaged, lengths)

                configs = [
                    ("A_singles", {"pair_top": 0, "triple_top": 0,
                     "plateau_threshold": 10 ** 9}),
                ]
                if n <= 16:
                    configs += [
                        ("B_pairs_full", {
                         "pair_top": B, "triple_top": 0, "plateau_threshold": 10 ** 9}),
                    ]
                configs += [
                    ("C_top16", {"pair_top": 16, "triple_top": 0,
                     "plateau_threshold": 10 ** 9}),
                    ("D_top32", {"pair_top": 32, "triple_top": 0,
                     "plateau_threshold": 10 ** 9}),
                    ("E_top64", {"pair_top": 64, "triple_top": 0,
                     "plateau_threshold": 10 ** 9}),
                    ("F_top128", {"pair_top": 128, "triple_top": 0,
                     "plateau_threshold": 10 ** 9}),
                ]

                for cfg_name, cfg_kw in configs:
                    rp = RepairExperiment(
                        n=n, sieve=False, verbose=False, **cfg_kw)
                    t0 = time.perf_counter()
                    result = rp.search(steps=step_budget, seed=trial_seed,
                                       sequences=damaged)
                    wall = time.perf_counter() - t0
                    row = {
                        "test": 4, "n": n, "d": d, "trial": trial,
                        "start_energy": e0, "end_energy": result.metrics["energy"],
                        "success": result.metrics["energy"] == 0,
                        "strategy": cfg_name, "wall_s": wall,
                        "hamming_to_original": hamming_distance(
                            result.sequences, sol, lengths),
                    }
                    _emit(out, row)

            print(f"    d={d:>3}: done")

        print(f"    => {out}")


# ── Test 5: Structured perturbations ──────────────────────────────────────────


def test5_structured(n_values: list[int], quick: bool, seed: int, out_dir: Path) -> None:
    """Errors bunched in one sequence, at edges, or spread evenly."""
    print("\n" + "=" * 60)
    print("TEST 5: Structured perturbations")
    print("=" * 60)

    for n in n_values:
        sol = tt_sequences(n)
        lengths = tt_lengths(n)
        B = int(lengths.sum())
        positions = _all_positions(lengths)
        out = out_dir / f"repair_test5_TT{n}.jsonl"
        n_trials = 3 if quick else 10

        d_values = [7, 14] if n >= 36 else [3, 5]
        print(f"  TT({n}) — {n_trials} trials each")

        for d in d_values:
            if d > B // 2:
                continue
            step_budget = max(120, 30 * d)
            for trial in range(n_trials):
                rng = np.random.default_rng(seed + trial)
                for ptype in ["one_seq_A", "bunched_A", "edges", "spread"]:
                    damaged = sol.copy()

                    if ptype == "one_seq_A":
                        # All flips in sequence A (spill to B if needed)
                        a_pos = [i for i, (r, c) in enumerate(
                            positions) if r == 0]
                        if d <= len(a_pos):
                            idxs = rng.choice(a_pos, size=d, replace=False)
                        else:
                            idxs = list(rng.choice(
                                a_pos, size=len(a_pos), replace=False))
                            b_pos = [i for i, (r, c) in enumerate(
                                positions) if r == 1]
                            idxs += list(rng.choice(b_pos, size=d -
                                         len(a_pos), replace=False))
                    elif ptype == "bunched_A":
                        # d consecutive bits in sequence A
                        start = rng.integers(0, int(lengths[0]) - d)
                        idxs = [i for i, (r, c) in enumerate(
                            positions) if r == 0 and start <= c < start + d]
                    elif ptype == "edges":
                        # Half at start of A, half at end of D
                        n_half = d // 2
                        a_start = [i for i, (r, c) in enumerate(
                            positions) if r == 0 and c < n_half]
                        d_end = [i for i, (r, c) in enumerate(
                            positions) if r == 3 and c >= int(lengths[3]) - (d - n_half)]
                        idxs = list(rng.choice(a_start, size=min(
                            n_half, len(a_start)), replace=False))
                        idxs += list(rng.choice(d_end, size=min(d -
                                     n_half, len(d_end)), replace=False))
                    else:  # spread
                        # Evenly across all 4 sequences
                        per_seq = max(1, d // 4)
                        idxs = []
                        for seq in range(4):
                            s_pos = [i for i, (r, c) in enumerate(
                                positions) if r == seq]
                            n_pick = min(per_seq, len(s_pos))
                            idxs += list(rng.choice(s_pos,
                                         size=n_pick, replace=False))
                        # Top up to exact d
                        while len(idxs) < d:
                            extra = rng.choice(B, size=1, replace=False)
                            if int(extra) not in idxs:
                                idxs.append(int(extra))

                    for idx in idxs:
                        r, c = positions[int(idx)]
                        damaged[r, c] *= -1

                    e0 = _exact_energy(damaged, lengths)
                    rp = RepairExperiment(n=n, sieve=False, pair_top=64,
                                          triple_top=0, plateau_threshold=10 ** 9,
                                          verbose=False)
                    t0 = time.perf_counter()
                    result = rp.search(steps=step_budget, seed=seed + trial,
                                       sequences=damaged)
                    wall = time.perf_counter() - t0
                    row = {
                        "test": 5, "n": n, "d": d, "trial": trial,
                        "perturbation_type": ptype, "start_energy": e0,
                        "end_energy": result.metrics["energy"],
                        "success": result.metrics["energy"] == 0,
                        "wall_s": wall,
                        "hamming_to_original": hamming_distance(
                            result.sequences, sol, lengths),
                    }
                    _emit(out, row)

            print(f"    d={d:>3}: done")

        print(f"    => {out}")


# ── Test 6: Pair utility ──────────────────────────────────────────────────────


def test6_pair_utility(n_values: list[int], quick: bool, seed: int, out_dir: Path) -> None:
    """Singles to stall, then measure pair existence and top-K coverage."""
    print("\n" + "=" * 60)
    print("TEST 6: Pair utility (singles to stall -> pair escape)")
    print("=" * 60)

    for n in n_values:
        sol = tt_sequences(n)
        lengths = tt_lengths(n)
        B = int(lengths.sum())
        positions = _all_positions(lengths)
        out = out_dir / f"repair_test6_TT{n}.jsonl"
        n_probes = 3 if quick else 5
        d_values = [2, 3, 4, 5, 7, 10]
        d_values = [d for d in d_values if d <= B // 3]

        print(f"  TT({n}) — {n_probes} probes x {len(d_values)} distances")
        rng = np.random.default_rng(seed)

        for d in d_values:
            for probe in range(n_probes):
                # Damage and run singles-only to stall
                damaged = _damage_random(rng, sol, positions, d)

                rp = RepairExperiment(n=n, sieve=False, pair_top=0, triple_top=0,
                                      plateau_threshold=10 ** 9, verbose=False)
                t0 = time.perf_counter()
                result_singles = rp.search(steps=100, seed=seed + probe,
                                           sequences=damaged)
                stall_e = result_singles.metrics["energy"]
                stall_seq = result_singles.sequences
                stall_state = stall_seq if result_singles.metrics.get(
                    "energy", 0) > 0 else damaged

                if stall_e == 0:
                    row = {"test": 6, "n": n, "d": d, "probe": probe,
                           "singles_solved": True, "wall_s": time.perf_counter() - t0}
                    _emit(out, row)
                    continue

                # Compute full exact pair scan at stall state
                stall_f = xp.asarray(stall_state.astype(
                    np.float32), dtype=xp.float32)
                deltas, r0_cpu = rp._single_deltas(stall_f, positions)
                stall_energies = np.sum((r0_cpu + deltas) ** 2, axis=1)
                top16 = np.argsort(stall_energies)[:16]
                top32 = np.argsort(stall_energies)[:32]
                top64 = np.argsort(stall_energies)[:64]

                # Exact: compute residual for every pair (expensive for large n)
                if n <= 16:
                    all_pairs = list(combinations(range(B), 2))
                else:
                    all_pairs = list(combinations(rng.choice(
                        B, size=min(B, 50), replace=False), 2))

                n_improving = 0
                best_pair_energy = stall_e
                best_pair_tuple = None
                for i, j in all_pairs:
                    dp = stall_state.copy()
                    ri, ci = positions[i]
                    rj, cj = positions[j]
                    dp[ri, ci] *= -1
                    dp[rj, cj] *= -1
                    e = _exact_energy(dp, lengths)
                    if e < best_pair_energy:
                        best_pair_energy = e
                        best_pair_tuple = (i, j)
                        n_improving += 1

                # Which top-K covers it?
                in_top16 = in_top32 = in_top64 = False
                if best_pair_tuple is not None:
                    bi, bj = best_pair_tuple
                    in_top16 = bi in top16 and bj in top16
                    in_top32 = bi in top32 and bj in top32
                    in_top64 = bi in top64 and bj in top64

                # Accept the best pair and continue with singles
                pairs_escape = False
                pairs_energy = None
                if best_pair_tuple is not None:
                    escaped_seq = stall_state.copy()
                    for idx in best_pair_tuple:
                        r, c = positions[idx]
                        escaped_seq[r, c] *= -1
                    rp2 = RepairExperiment(n=n, sieve=False, pair_top=0, triple_top=0,
                                           plateau_threshold=10 ** 9, verbose=False)
                    result2 = rp2.search(steps=100, seed=seed + probe + 1000,
                                         sequences=escaped_seq)
                    pairs_escape = result2.metrics["energy"] == 0
                    pairs_energy = result2.metrics["energy"]

                row = {
                    "test": 6, "n": n, "d": d, "probe": probe,
                    "singles_solved": False, "stall_energy": stall_e,
                    "n_improving_pairs": n_improving,
                    "pair_in_top16": in_top16, "pair_in_top32": in_top32,
                    "pair_in_top64": in_top64,
                    "pair_energy_drop": stall_e - best_pair_energy if best_pair_tuple else 0,
                    "pair_leads_to_solution": pairs_escape,
                    "pair_final_energy": pairs_energy,
                    "wall_s": time.perf_counter() - t0,
                }
                _emit(out, row)

            print(f"    d={d:>3}: done")

        print(f"    => {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair tests 1-6")
    parser.add_argument("test", type=int, choices=[1, 2, 3, 4, 5, 6], nargs="*",
                        default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--tt", type=str, default="8,16",
                        help="Comma-separated n values (default: 8,16)")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer trials for faster runs")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.tt.split(",")]
    out_dir = Path(args.out)

    all_tests = args.test
    tests = {
        1: lambda: test1_correctness(n_values, out_dir),
        2: lambda: test2_sensitivity_singles(n_values, out_dir),
        3: lambda: test3_sensitivity_pairs(n_values, args.quick, out_dir),
        4: lambda: test4_repair_radius(n_values, args.quick, args.seed, out_dir),
        5: lambda: test5_structured(n_values, args.quick, args.seed, out_dir),
        6: lambda: test6_pair_utility(n_values, args.quick, args.seed, out_dir),
    }

    for t in all_tests:
        tests[t]()


if __name__ == "__main__":
    main()
