"""Bounded full-scan prototype; no additional production configuration switch."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.figure import Figure

from lab.experiments.replay import reference_scan
from lab.recover import decode
from src import solver
from src.generator import RANDOM_START
from src.models import CandidateBudget, Int8Array, SearchStats
from src.tracker import Tracker
from src.verify import verify_candidate


def full_scan(
    cur_seq: Int8Array,
    tracker: Tracker,
    positions: tuple[tuple[int, int], ...],
    cur_e: int,
    budget: CandidateBudget,
    q_scale: int,
    cfg: solver.SolverConfig,
    stats: SearchStats,
) -> tuple[bool, int, int]:
    """Prioritize a direct solution, otherwise retain first-improvement ordering."""
    assert cfg.geo_weight == 0, "prototype is restricted to the default scoring policy"
    used = budget.take(len(positions))
    stats.greedy_candidate_evals += used
    energies = tracker.flip_batch(0, used)
    zero = np.flatnonzero(energies == 0)
    improving = np.flatnonzero(energies < cur_e)
    candidates = zero if zero.size else improving
    if not candidates.size:
        return False, cur_e, used
    index = int(candidates[0])
    updated = tracker.accept(cur_seq, *positions[index])
    stats.greedy_energy_improvement += cur_e - updated
    stats.hit_greedy()
    if updated == 0:
        stats.record_solve("greedy", cur_e // q_scale)
    return True, updated, used


def main() -> None:
    """Screen paired random starts and replay certified one-flip controls."""
    output = Path("runs/research/landscape")
    output.mkdir(parents=True, exist_ok=True)
    original = solver._greedy_descent
    baseline = reference_scan()
    results: list[dict[str, object]] = []
    try:
        for n in (36, 40, 44):
            for seed in range(42, 192):
                for label, function in (("baseline", baseline), ("full_scan", full_scan)):
                    solver._greedy_descent = function
                    rng = np.random.default_rng(seed)
                    state = RANDOM_START.build(n, rng)
                    started = perf_counter()
                    result = solver.search(state, Tracker(), rng, candidate_budget=6_000_000)
                    elapsed = perf_counter() - started
                    if result.solved:
                        verify_candidate(result.sequences)
                    results.append(
                        {
                            "n": n,
                            "seed": seed,
                            "arm": label,
                            "solved": result.solved,
                            "q": result.energy // (64 * n),
                            "evals": result.candidate_evals,
                            "elapsed": elapsed,
                        }
                    )
            print(
                f"n={n}: "
                + str(
                    {
                        arm: sum(
                            1 for r in results if r["n"] == n and r["arm"] == arm and r["solved"]
                        )
                        for arm in ("baseline", "full_scan")
                    }
                ),
                flush=True,
            )
            (output / "greedy-screen.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
        controls = json.loads((output / "recovery.json").read_text(encoding="utf-8"))
        replay: list[dict[str, object]] = []
        solver._greedy_descent = full_scan
        with sqlite3.connect("file:data/hadamard.db?mode=ro", uri=True) as connection:
            for control in controls:
                if control["radius"] != 1:
                    continue
                encoded = connection.execute(
                    "SELECT seqs_b64 FROM runs WHERE id=?", (control["parent_id"],)
                ).fetchone()[0]
                state = decode(encoded, control["n"])
                state.flat[control["flips"]] *= -1
                result = solver.search(
                    state,
                    Tracker(),
                    np.random.default_rng(control["seed"]),
                    candidate_budget=250_000,
                )
                if result.solved:
                    verify_candidate(result.sequences)
                replay.append(
                    {
                        "parent_id": control["parent_id"],
                        "n": control["n"],
                        "seed": control["seed"],
                        "baseline_q": control["end_q"],
                        "full_scan_q": result.energy // (64 * control["n"]),
                        "evals": result.candidate_evals,
                    }
                )
        (output / "greedy-recovery.json").write_text(json.dumps(replay, indent=2), encoding="utf-8")
    finally:
        solver._greedy_descent = original
    plot_results()


def plot_results() -> None:
    """Render retained recovery measurements without rerunning searches."""
    output = Path("runs/research/landscape")
    recovery = json.loads((output / "recovery.json").read_text(encoding="utf-8"))
    replay = json.loads((output / "greedy-recovery.json").read_text(encoding="utf-8"))
    figure = Figure(figsize=(12, 5), layout="constrained")
    curve, compare = figure.subplots(1, 2)
    lengths = sorted({r["n"] for r in recovery})
    radii = (1, 4, 8, 16, 32)
    for n in lengths:
        rates: list[float] = []
        for radius in radii:
            rows = [r for r in recovery if r["n"] == n and r["radius"] == radius]
            rates.append(100 * sum(r["end_q"] == 0 for r in rows) / len(rows))
        curve.plot(radii, rates, "o-", label=f"n={n}")
    curve.set(
        title="Recovery within 250k evaluations",
        xlabel="Bits flipped from a known solution",
        ylabel="Solved perturbations (%)",
        ylim=(0, 105),
    )
    curve.set_xticks(radii)
    curve.legend()
    x = np.arange(len(lengths))
    for offset, key, label in (
        (-0.18, "baseline_q", "First improvement"),
        (0.18, "full_scan_q", "Full scan"),
    ):
        rates = []
        for n in lengths:
            rows = [r for r in replay if r["n"] == n]
            rates.append(100 * sum(r[key] == 0 for r in rows) / len(rows))
        compare.bar(x + offset, rates, width=0.36, label=label)
    compare.set_xticks(x, [str(n) for n in lengths])
    compare.set(
        title="One-bit controls: 220 total",
        xlabel="Sequence length n",
        ylabel="Solved (%)",
        ylim=(0, 105),
    )
    compare.legend(loc="lower left")
    # Public Matplotlib kwargs have incomplete third-party annotations.
    figure.suptitle(  # pyright: ignore[reportUnknownMemberType]
        "Local recoverability is not random-start search yield\n"
        "3 parents at n=36/40/44; 1 parent at n=48/52. Repeats are dependent."
    )
    for suffix in ("png", "svg"):
        figure.savefig(output / f"recovery.{suffix}", dpi=160)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    main()
