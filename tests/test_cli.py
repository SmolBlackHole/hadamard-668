"""CLI integration tests — keep `--workers 1` to avoid process storms."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest

import run
from run import _execute_single, _format_time
from src.generator import RANDOM_START, StartConstruction
from src.models import RunResult, SearchStats
from src.solver import SolverConfig


def test_run_help_works() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--symmetric-bs" not in result.stdout
    assert "--candidate-budget" in result.stdout
    assert "SQLite" in result.stdout


def test_format_time() -> None:
    assert _format_time(0.0005) == "500us"
    assert _format_time(0.5) == "500ms"
    assert _format_time(5.0) == "5.0s"


def test_execute_single_smoke() -> None:
    result = _execute_single("gs4", 5, 200_000, 42, SolverConfig(), RANDOM_START)
    assert result.solved


def test_execute_single_paley_ng_smoke() -> None:
    result = _execute_single("paley-ng", 52, 1, 42, SolverConfig(), RANDOM_START)
    assert result.solved
    assert result.order == 208


def test_execute_single_construct_smoke() -> None:
    result = _execute_single("construct", 104, 1, 42, SolverConfig(), RANDOM_START)
    assert result.solved
    assert result.order == 416


def test_parallel_progress_is_compact_and_preserves_result_order(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class ImmediatePool:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def submit(self, function: Any, *args: object) -> Future[RunResult]:
            future: Future[RunResult] = Future()
            try:
                future.set_result(function(*args))
            except Exception as error:
                future.set_exception(error)
            return future

    def fake_execute(
        strategy: str,
        n: int,
        candidate_budget: int,
        seed: int,
        _config: SolverConfig,
        _start: StartConstruction,
    ) -> RunResult:
        if seed == 2:
            raise RuntimeError("boom")
        energy = 0 if seed == 1 else 64 * n
        return RunResult(
            strategy=strategy,
            n=n,
            seed=seed,
            sequences=np.ones((4, n), dtype=np.int8),
            energy=energy,
            candidate_evals=candidate_budget,
            elapsed_seconds=1.0,
            stats=SearchStats(),
            verified=energy == 0,
        )

    def reverse_futures(
        futures: Iterable[Future[RunResult]],
    ) -> Iterator[Future[RunResult]]:
        return reversed(list(futures))

    times = iter((0.0, 0.6, 1.2, 1.8))
    monkeypatch.setattr(run, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(run, "as_completed", reverse_futures)
    monkeypatch.setattr(run, "_execute_single", fake_execute)
    monkeypatch.setattr(run.time, "perf_counter", lambda: next(times))
    tasks = [("gs4", 5, 100, seed, SolverConfig(), RANDOM_START) for seed in (1, 2, 3)]

    results = run._execute_all(tasks, workers=2)
    output = capsys.readouterr().out

    assert [result.seed for result in results] == [1, 3]
    assert "[3/3] solved=1 failed=1 bestQ=0" in output
    assert "seed=2 FAILED: boom" in output


def test_worker_ignores_sigint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[signal.Signals, signal.Handlers]] = []

    def record_signal(sig: signal.Signals, handler: signal.Handlers) -> signal.Handlers:
        calls.append((sig, handler))
        return signal.SIG_DFL

    monkeypatch.setattr(run.signal, "signal", record_signal)

    run._ignore_sigint()

    assert calls == [(signal.SIGINT, signal.SIG_IGN)]


def test_run_plan_and_save_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "runs.db"
    config = SolverConfig()
    result = RunResult(
        strategy="gs4",
        n=1,
        seed=7,
        sequences=np.ones((4, 1), dtype=np.int8),
        energy=0,
        candidate_evals=100,
        elapsed_seconds=0.1,
        stats=SearchStats(),
        verified=True,
        construction="random",
        candidate_budget=100,
    )
    tasks = [("gs4", 1, 100, 7, config, RANDOM_START)]

    run._print_run_plan(tasks, workers=1, output=path)
    run._save_all(path, [result])
    output = capsys.readouterr().out

    assert "Search: strategy=gs4 n=1 order=4 runs=1 seeds=7" in output
    assert "budget=100 candidate evaluations/run workers=1" in output
    assert f"Saving 1 run to {path}" in output
    assert "DB write [1/1]" in output
