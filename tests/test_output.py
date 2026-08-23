"""Tests for output persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import src.output as output
from src.models import RunResult, SearchStats
from src.output import load_runs, save_run, save_runs


def _make_result(
    seed: int,
    energy: int,
    elapsed: float = 1.0,
    candidate_evals: int = 100,
    stats: SearchStats | None = None,
) -> RunResult:
    seqs = np.ones((4, 6), dtype=np.int8)
    seqs[0, seed % seqs.shape[1]] = -1
    return RunResult(
        strategy="gs4",
        n=6,
        seed=seed,
        sequences=seqs,
        energy=energy,
        candidate_evals=candidate_evals,
        elapsed_seconds=elapsed,
        stats=stats or SearchStats(),
        verified=energy == 0,
    )


def test_save_and_load_solution(tmp_path: Path) -> None:
    stats = SearchStats()
    stats.greedy_moves = 3
    stats.greedy_streaks = [3]

    path = tmp_path / "runs.db"
    save_run(path, _make_result(42, energy=0, stats=stats))

    data = load_runs(path)
    runs = data["gs4"]["6"]
    assert len(runs) == 1
    r = runs[0]
    assert r["seed"] == 42
    assert r["solved"]
    assert r["energy"] == 0
    assert r["elapsed_s"] == 1.0
    assert r["candidate_evals"] == 100
    assert r["stats"]["stats_schema_version"] == 2
    assert r["stats"]["greedy_moves"] == 3
    assert len(r["sha256"]) == 64
    assert r["sha256"] == r["validation_hash"]
    assert r["class"] == r["orbit_hash"]
    assert r["canonicalizer"] == "gs4-quick-orbit-v1"
    assert r["valid"]
    assert len(r["seqs_b64"]) > 0
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT iterations FROM runs").fetchone() == (0,)


def test_save_and_load_failure(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    save_run(path, _make_result(1, energy=16, elapsed=3.0, candidate_evals=5000))

    data = load_runs(path)
    r = data["gs4"]["6"][0]
    assert not r["solved"]
    assert r["energy"] == 16
    assert r["elapsed_s"] == 3.0
    assert r["candidate_evals"] == 5000


def test_load_empty_dataset(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "nonexistent.db") == {}


def test_multiple_runs_same_n(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    save_run(path, _make_result(0, energy=0, elapsed=1.0, candidate_evals=100))
    save_run(path, _make_result(1, energy=4, elapsed=2.0, candidate_evals=200))

    data = load_runs(path)
    runs = data["gs4"]["6"]
    assert len(runs) == 2
    assert [r["seed"] for r in runs] == [0, 1]
    assert [r["solved"] for r in runs] == [True, False]


def test_save_runs_uses_one_revision_and_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "runs.db"
    revision_calls = 0
    progress: list[tuple[int, int]] = []

    def revision() -> str:
        nonlocal revision_calls
        revision_calls += 1
        return "test-revision"

    monkeypatch.setattr(output, "_code_revision", revision)
    save_runs(
        path,
        [_make_result(0, energy=0), _make_result(1, energy=4)],
        lambda saved, total: progress.append((saved, total)),
    )

    runs = load_runs(path)["gs4"]["6"]
    assert revision_calls == 1
    assert progress == [(1, 2), (2, 2)]
    assert [run["seed"] for run in runs] == [0, 1]
    assert [run["code_revision"] for run in runs] == ["test-revision", "test-revision"]


def test_identical_states_from_different_seeds_are_retained(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    save_run(path, _make_result(0, energy=4))
    save_run(path, _make_result(6, energy=4))

    runs = load_runs(path)["gs4"]["6"]
    assert [run["seed"] for run in runs] == [0, 6]
    assert runs[0]["sha256"] == runs[1]["sha256"]


def test_repeated_seed_retains_experimental_history(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    save_run(path, _make_result(4, energy=16, candidate_evals=100))
    save_run(path, _make_result(4, energy=8, candidate_evals=200))

    runs = load_runs(path)["gs4"]["6"]
    assert len(runs) == 2
    assert [run["energy"] for run in runs] == [16, 8]
    assert [run["candidate_evals"] for run in runs] == [100, 200]
