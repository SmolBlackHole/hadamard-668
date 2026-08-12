from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.q1_diagnosis import analyze_q1_state, diagnose_database
from src.models import RunResult, SearchStats
from src.output import load_runs, save_run
from src.tracker import Tracker


def _q1_state(n: int = 5) -> np.ndarray:
    rng = np.random.default_rng(668)
    for _ in range(10_000):
        sequences = rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, n))
        tracker = Tracker()
        tracker.build(sequences)
        if tracker._q == 1:
            return sequences
    raise AssertionError("failed to find a deterministic Q=1 fixture")


def test_q1_diagnosis_exhaustively_matches_tracker() -> None:
    sequences = _q1_state()
    n = sequences.shape[1]
    tracker = Tracker()
    tracker.build(sequences)

    report = analyze_q1_state(sequences)

    single_qs = tracker.flip_qs()
    pair_qs = [
        tracker.combo_energy([divmod(left, n), divmod(right, n)]) // (64 * n)
        for left, right in combinations(range(4 * n), 2)
    ]
    assert report["best_single_q"] == int(single_qs.min())
    assert report["n_exact_single"] == int(np.count_nonzero(single_qs == 0))
    assert report["target_delta_present"] == bool(np.any(single_qs == 0))
    assert report["best_pair_q"] == min(pair_qs)
    assert report["n_solving_pairs"] == pair_qs.count(0)
    assert report["single_flip_local_minimum"] == bool(np.all(single_qs >= 1))
    assert report["two_flip_local_minimum"] == (min(pair_qs) >= 1)
    expected_depth = 1 if np.any(single_qs == 0) else 2 if 0 in pair_qs else ">2"
    assert report["repair_depth"] == expected_depth


def test_database_diagnosis_reads_persisted_q1_endpoint(tmp_path: Path) -> None:
    sequences = _q1_state()
    n = sequences.shape[1]
    tracker = Tracker()
    tracker.build(sequences)
    result = RunResult(
        strategy="gs4",
        n=n,
        seed=17,
        sequences=sequences,
        energy=tracker.energy(),
        steps=0,
        elapsed_seconds=1.0,
        stats=SearchStats(),
        verified=False,
    )
    path = tmp_path / "runs.db"
    save_run(path, result)

    report = diagnose_database(path, selected_n={n})

    summary = report["summary"]
    assert isinstance(summary, dict)
    assert summary[str(n)]["q1_endpoints"] == 1
    states = report["q1_states"]
    assert isinstance(states, list)
    assert states[0]["seed"] == 17
    assert load_runs(path)["gs4"][str(n)][0]["solver_e"] == tracker.energy()
