"""Tests for the repository quality command runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import quality


def test_check_runs_every_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(quality, "ROOT", tmp_path)

    def fake_run(
        command: tuple[str, ...], *, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        assert cwd == quality.ROOT
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(quality.subprocess, "run", fake_run)

    assert quality.main(["check"]) == 0
    assert calls == list(quality.COMMANDS["check"])
    assert (tmp_path / "runs").is_dir()


def test_runner_stops_at_first_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_run(
        command: tuple[str, ...], *, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(quality.subprocess, "run", fake_run)

    assert quality.main(["check"]) == 7
    assert calls == 1
