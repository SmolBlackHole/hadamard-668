"""Tests for output persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from generator import Result
from metrics import Metrics
from output import load_runs, save_run
from solver import SearchStats


def _make_result(
    seed: int,
    energy: int,
    elapsed: float = 1.0,
    iterations: int = 100,
    stats: SearchStats | None = None,
) -> Result:
    seqs = np.ones((4, 6), dtype=np.int8)
    if energy == 0:
        seqs[0, 0] = -1
    order = 24
    matrix = np.ones((order, order), dtype=np.int8) if energy > 0 else _hadamard_24(seqs)
    return Result(
        matrix=matrix,
        metrics=Metrics(energy=energy, orthogonal_pairs=0, max_abs_correlation=0),
        elapsed=elapsed,
        seed=seed,
        iterations=iterations,
        sequences=seqs,
        stats=stats,
    )


def _hadamard_24(seqs: np.ndarray) -> np.ndarray:
    """Build a real Hadamard matrix from sequences for testing."""
    from builder import Builder

    b = Builder(kind="gs4", n=6)
    return b.build(seqs)


def test_save_and_load_solution(tmp_path: Path) -> None:
    stats = SearchStats()
    stats.singles = 3
    stats.pairs = 1
    stats.singles_streaks = [3]
    stats._flush()

    path = tmp_path / "runs.json"
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
    assert r["stats"]["pairs"] == 1
    assert r["stats"]["singles_streaks"] == [3]
    assert len(r["sha256"]) == 64
    assert len(r["seqs_b64"]) > 0


def test_save_and_load_failure(tmp_path: Path) -> None:
    path = tmp_path / "runs.json"
    save_run(path, "gs4", 6, _make_result(1, energy=16, elapsed=3.0, iterations=5000))

    data = load_runs(path)
    r = data["gs4"]["6"][0]
    assert not r["solved"]
    assert r["energy"] == 16
    assert r["elapsed_s"] == 3.0
    assert r["iterations"] == 5000


def test_load_empty_dataset(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "nonexistent.json") == {}


def test_multiple_runs_same_n(tmp_path: Path) -> None:
    path = tmp_path / "runs.json"
    save_run(path, "gs4", 6, _make_result(0, energy=0, elapsed=1.0, iterations=100))
    save_run(path, "gs4", 6, _make_result(1, energy=4, elapsed=2.0, iterations=200))

    data = load_runs(path)
    runs = data["gs4"]["6"]
    assert len(runs) == 2
    assert [r["seed"] for r in runs] == [0, 1]
    assert [r["solved"] for r in runs] == [True, False]
