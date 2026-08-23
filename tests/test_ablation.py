from __future__ import annotations

import pytest

from scripts.ablation import _aggregate


def test_aggregate_reports_total_effort_and_conditional_solve_latency() -> None:
    runs = [
        {
            "solved": True,
            "energy": 0,
            "elapsed": 2.0,
            "candidate_evals": 100,
            "validation_hash": "exact-a",
            "orbit_hash": "orbit-a",
            "stats": {
                "greedy_moves": 2,
                "total_accepted_moves": 2,
                "solve_phase": "greedy",
            },
        },
        {
            "solved": False,
            "energy": 64,
            "elapsed": 3.0,
            "candidate_evals": 200,
            "validation_hash": None,
            "orbit_hash": None,
            "stats": {
                "greedy_moves": 4,
                "total_accepted_moves": 4,
                "solve_phase": None,
            },
        },
    ]

    aggregate = _aggregate(runs)

    assert aggregate["solved"] == 1
    assert aggregate["total"] == 2
    assert aggregate["solve_rate"] == 0.5
    assert aggregate["seconds_per_solution"] == 5.0
    assert aggregate["solutions_per_worker_hour"] == 720.0
    assert aggregate["candidate_evals_per_solve"] == 300
    assert aggregate["solved_candidate_evals_p50"] == 100
    assert aggregate["solved_candidate_evals_p90"] == 100
    assert aggregate["unique_exact_solutions"] == 1
    assert aggregate["unique_orbits"] == 1
    assert aggregate["greedy_moves"] == 6
    assert aggregate["solve_phases"] == {"greedy": 1}


def test_aggregate_rejects_no_runs() -> None:
    with pytest.raises(ValueError, match="runs cannot be empty"):
        _aggregate([])
