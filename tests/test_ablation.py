from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lab.compare import _aggregate, main


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


@pytest.mark.parametrize("definitions", ["{}", "[]", '{"bad": {"not_a_setting": 1}}'])
def test_custom_configs_reject_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    definitions: str,
) -> None:
    source = tmp_path / "configs.json"
    source.write_text(definitions)
    monkeypatch.setattr(sys, "argv", ["ablation", "--configs", str(source)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_quench_ablation_pairs_seeds_and_preserves_total_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "quench.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ablation",
            "--quench-budget",
            "2",
            "--quench-budget",
            "2",
            "--n",
            "4",
            "--seeds",
            "2",
            "--seed-start",
            "10",
            "--candidate-budget",
            "8",
            "--workers",
            "1",
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert set(report["configs"]) == {"default", "no-targeted", "quench=2"}
    assert report["seed_start"] == 10
    assert len(report["runs"]) == 6
    for name in report["configs"]:
        runs = [r for r in report["runs"] if r["config_name"] == name]
        assert [r["seed"] for r in runs] == [10, 11]
        assert all(r["candidate_evals"] <= 8 for r in runs)
