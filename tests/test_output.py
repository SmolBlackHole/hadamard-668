"""Tests for output persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import output
from src.generator import Result
from src.metrics import Metrics
from src.output import (
    load_runs,
    load_symmetric_bs_runs,
    save_run,
    save_symmetric_bs_result,
)
from src.solver import SearchStats
from src.symmetric_bs_searcher import PROFILES, SearchConfig, search_profile


def _make_result(
    seed: int,
    energy: int,
    elapsed: float = 1.0,
    iterations: int = 100,
    stats: SearchStats | None = None,
) -> Result:
    seqs = np.ones((4, 6), dtype=np.int8)
    seqs[0, seed % seqs.shape[1]] = -1
    return Result(
        matrix=np.empty((0, 0), dtype=np.int8),
        metrics=Metrics(energy=energy, orthogonal_pairs=0, max_abs_correlation=0),
        elapsed=elapsed,
        seed=seed,
        iterations=iterations,
        sequences=seqs,
        stats=stats,
    )


def test_save_and_load_solution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(output, "SOLUTIONS", tmp_path / "solutions.json")
    stats = SearchStats()
    stats.singles = 3
    stats.singles_streaks = [3]

    path = tmp_path / "runs.db"
    save_run(path, "gs4", 6, _make_result(42, energy=0, stats=stats))

    data = load_runs(path)
    runs = data["gs4"]["6"]
    assert len(runs) == 1
    r = runs[0]
    assert r["seed"] == 42
    assert r["solved"]
    assert r["energy"] == 0
    assert r["elapsed_s"] == 1.0
    assert r["iterations"] == 100
    assert r["stats"]["singles"] == 3
    assert len(r["sha256"]) == 64
    assert len(r["seqs_b64"]) > 0


def test_save_and_load_failure(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    save_run(path, "gs4", 6, _make_result(1, energy=16, elapsed=3.0, iterations=5000))

    data = load_runs(path)
    r = data["gs4"]["6"][0]
    assert not r["solved"]
    assert r["energy"] == 16
    assert r["elapsed_s"] == 3.0
    assert r["iterations"] == 5000


def test_load_empty_dataset(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "nonexistent.db") == {}


def test_multiple_runs_same_n(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(output, "SOLUTIONS", tmp_path / "solutions.json")
    path = tmp_path / "runs.db"
    save_run(path, "gs4", 6, _make_result(0, energy=0, elapsed=1.0, iterations=100))
    save_run(path, "gs4", 6, _make_result(1, energy=4, elapsed=2.0, iterations=200))

    data = load_runs(path)
    runs = data["gs4"]["6"]
    assert len(runs) == 2
    assert [r["seed"] for r in runs] == [0, 1]
    assert [r["solved"] for r in runs] == [True, False]


def test_identical_states_from_different_seeds_are_retained(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    save_run(path, "gs4", 6, _make_result(0, energy=4))
    save_run(path, "gs4", 6, _make_result(6, energy=4))

    runs = load_runs(path)["gs4"]["6"]
    assert [run["seed"] for run in runs] == [0, 6]
    assert runs[0]["sha256"] == runs[1]["sha256"]


def test_repeated_seed_replaces_its_previous_run(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    save_run(path, "gs4", 6, _make_result(4, energy=16, iterations=100))
    save_run(path, "gs4", 6, _make_result(4, energy=8, iterations=200))

    runs = load_runs(path)["gs4"]["6"]
    assert len(runs) == 1
    assert runs[0]["energy"] == 8
    assert runs[0]["iterations"] == 200


def test_symmetric_bs_result_round_trips_through_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    result = search_profile(PROFILES[0], SearchConfig(restarts=1, steps=0, pair_top_k=0), seed=42)

    config = SearchConfig(restarts=1, steps=0, pair_top_k=0)
    save_symmetric_bs_result(path, result, base_seed=42, config=config)

    stored = load_symmetric_bs_runs(path)
    assert len(stored) == 1
    assert stored[0]["profile"] == list(PROFILES[0].as_tuple())
    assert stored[0]["seed"] == result.seed
    assert stored[0]["base_seed"] == 42
    assert stored[0]["config"]["steps"] == 0
    assert stored[0]["best_energy"] == result.best_energy
    assert stored[0]["residual"] == result.residual.tolist()
