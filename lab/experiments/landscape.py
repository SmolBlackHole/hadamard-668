"""Compare solution-derived and stored search states at identical n and Q.

This bounded diagnostic is not a solver benchmark or a training dataset. Controls
share parents; continuation repetitions share starts. Treat state/parent clusters
as dependent. Generated states and all continuation results are retained locally.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from lab.experiments.replay import reference_scan
from lab.recover import decode, descend, geometry
from src import solver
from src.models import Int8Array
from src.solver import search
from src.tracker import Tracker
from src.verify import verify_candidate


def continuation(state: Int8Array) -> list[dict[str, int]]:
    """Run three identical-budget searches with fresh, recorded RNG seeds."""
    results: list[dict[str, int]] = []
    for seed in (200001, 200002, 200003):
        result = search(
            state.copy(), Tracker(), np.random.default_rng(seed), candidate_budget=250_000
        )
        if result.solved:
            verify_candidate(result.sequences)
        results.append(
            {
                "seed": seed,
                "q": result.energy // (64 * state.shape[1]),
                "evals": result.candidate_evals,
            }
        )
    return results


def main() -> None:
    """Generate controls, match endpoints and persist exact source states/results."""
    output = Path("runs/research/landscape")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    with sqlite3.connect("file:data/hadamard.db?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT id,n,seed,energy,seqs_b64,solved FROM runs "
                "WHERE id BETWEEN 6301 AND 20300 AND valid=1 ORDER BY id"
            )
        ]
    reports: list[dict[str, Any]] = []
    for n in (36, 40, 44, 48, 52):
        selected = [row for row in rows if row["n"] == n]
        parents = list({row["seqs_b64"]: row for row in selected if row["solved"]}.values())[:3]
        failures = [row for row in selected if not row["solved"]]
        controls: dict[int, list[dict[str, Any]]] = defaultdict(list)
        seen: set[bytes] = set()
        direct_q: Counter[int] = Counter()
        descent_q: Counter[int] = Counter()
        for parent in parents:
            original = decode(parent["seqs_b64"], n)
            verify_candidate(original)
            for attempt in range(800):
                radius = (1, 2, 4, 8)[attempt % 4]
                candidate = original.copy()
                candidate.flat[rng.choice(4 * n, radius, replace=False)] *= -1
                tracker = Tracker()
                tracker.build(candidate)
                direct_q[tracker.energy() // (64 * n)] += 1
                method = "direct" if attempt % 8 < 4 else "descent"
                if method == "descent":
                    candidate = descend(candidate)
                    tracker.build(candidate)
                    descent_q[tracker.energy() // (64 * n)] += 1
                q = tracker.energy() // (64 * n)
                if q == 0 or candidate.tobytes() in seen:
                    continue
                seen.add(candidate.tobytes())
                controls[q].append(
                    {
                        "parent_id": parent["id"],
                        "method": method,
                        "distance_to_parent": int(np.sum(candidate != original)),
                        "seqs_b64": base64.b64encode(candidate.tobytes()).decode(),
                    }
                )
        rng.shuffle(failures)
        pairs: list[dict[str, Any]] = []
        for failure in failures:
            q = failure["energy"] // (64 * n)
            if not controls[q]:
                continue
            control = controls[q].pop()
            pair: dict[str, Any] = {
                "n": n,
                "q": q,
                "failure_id": failure["id"],
                "control": control,
                "failure_seqs_b64": failure["seqs_b64"],
            }
            for label, encoded in (
                ("control", control["seqs_b64"]),
                ("failure", failure["seqs_b64"]),
            ):
                state = decode(encoded, n)
                pair[label + "_geometry"] = geometry(state)
                pair[label + "_continuations"] = continuation(state)
            pairs.append(pair)
            if len(pairs) == 20:
                break
        report = {
            "n": n,
            "parents": [p["id"] for p in parents],
            "direct_q": dict(sorted(direct_q.items())),
            "descent_q": dict(sorted(descent_q.items())),
            "pairs": pairs,
        }
        reports.append(report)
        print(
            f"n={n}: {len(parents)} parents, {len(pairs)} matched pairs; "
            f"control solves={sum(r['q'] == 0 for p in pairs for r in p['control_continuations'])}; "
            f"failure solves={sum(r['q'] == 0 for p in pairs for r in p['failure_continuations'])}",
            flush=True,
        )
        (output / "matched.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


def recovery() -> None:
    """Measure recovery versus certified perturbation radius, without Q matching."""
    output = Path("runs/research/landscape")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260908)
    records: list[dict[str, Any]] = []
    with sqlite3.connect("file:data/hadamard.db?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT id,n,seqs_b64 FROM runs WHERE id BETWEEN 6301 AND 20300 "
                "AND valid=1 AND solved=1 ORDER BY id"
            )
        ]
    for n in (36, 40, 44, 48, 52):
        parents = list({row["seqs_b64"]: row for row in rows if row["n"] == n}.values())[:3]
        for parent in parents:
            original = decode(parent["seqs_b64"], n)
            verify_candidate(original)
            for radius in (1, 4, 8, 16, 32):
                for repeat in range(20):
                    state = original.copy()
                    flips = rng.choice(4 * n, radius, replace=False)
                    state.flat[flips] *= -1
                    tracker = Tracker()
                    tracker.build(state)
                    q = tracker.energy() // (64 * n)
                    seed = 300000 + repeat
                    result = search(
                        state, Tracker(), np.random.default_rng(seed), candidate_budget=250_000
                    )
                    if result.solved:
                        verify_candidate(result.sequences)
                    records.append(
                        {
                            "n": n,
                            "parent_id": parent["id"],
                            "radius": radius,
                            "flips": flips.tolist(),
                            "seed": seed,
                            "start_q": q,
                            "end_q": result.energy // (64 * n),
                            "evals": result.candidate_evals,
                        }
                    )
        selected = [row for row in records if row["n"] == n]
        print(
            f"Recovery n={n}: "
            + str(
                {
                    radius: sum(r["end_q"] == 0 for r in selected if r["radius"] == radius)
                    for radius in (1, 4, 8, 16, 32)
                }
            ),
            flush=True,
        )
        (output / "recovery.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    original_scan = solver._greedy_descent
    solver._greedy_descent = reference_scan()
    try:
        main()
        recovery()
    finally:
        solver._greedy_descent = original_scan
