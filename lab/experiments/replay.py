"""Replay selected n=56 seeds with the master and research solver implementations."""

from __future__ import annotations

import ast
import base64
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src import solver
from src.generator import RANDOM_START
from src.tracker import Tracker


def reference_scan() -> Callable[..., tuple[bool, int, int]]:
    """Load the recorded pre-change scan so reruns keep the original baseline."""
    source = subprocess.check_output(["git", "show", "bb26d1d:src/solver.py"], text=True)
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_greedy_descent"
    )
    namespace = dict(vars(solver))
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "bb26d1d:src/solver.py", "exec"),
        namespace,
    )
    return namespace["_greedy_descent"]


def main() -> None:
    """Replay the original experiment with its fixed pre-full-scan policy."""
    original = solver._greedy_descent
    solver._greedy_descent = reference_scan()
    try:
        replay()
    finally:
        solver._greedy_descent = original


def replay() -> None:
    """Compare exact output bytes, preserving source and stored batch provenance."""
    revision = "bb26d1dfd87e8e533a5de2c3b98986071f851a85"
    source = subprocess.check_output(["git", "show", f"{revision}:src/solver.py"], text=True)
    output = Path("runs/research/landscape")
    output.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="gs4-replay-") as directory:
        path = Path(directory) / "baseline_solver.py"
        path.write_text(source, encoding="utf-8")
        name = "src._replay_baseline_solver"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[name] = baseline
        spec.loader.exec_module(baseline)
        with sqlite3.connect("file:data/hadamard.db?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            pairs = connection.execute(
                "SELECT a.seed,a.seqs_b64 AS old,b.seqs_b64 AS new,"
                "a.candidate_budget AS old_budget,b.candidate_budget AS new_budget FROM runs a JOIN runs b "
                "ON a.n=b.n AND a.seed=b.seed WHERE a.id BETWEEN 6301 AND 20300 "
                "AND b.id BETWEEN 20301 AND 28300 AND a.n=56 ORDER BY a.seed LIMIT 32"
            ).fetchall()
        for row in pairs:
            states: dict[str, str] = {}
            for label, module in (("master", baseline), ("research", solver)):
                rng = np.random.default_rng(row["seed"])
                start = RANDOM_START.build(56, rng)
                result = module.search(start, Tracker(), rng, candidate_budget=row["old_budget"])
                states[label] = base64.b64encode(result.sequences.tobytes()).decode()
            rng = np.random.default_rng(row["seed"])
            start = RANDOM_START.build(56, rng)
            actual = solver.search(start, Tracker(), rng, candidate_budget=row["new_budget"])
            states["actual"] = base64.b64encode(actual.sequences.tobytes()).decode()
            record: dict[str, object] = {
                "seed": row["seed"],
                "implementations_equal": states["master"] == states["research"],
                "master_reproduced": states["master"] == row["old"],
                "research_reproduced": states["actual"] == row["new"],
                "old_budget": row["old_budget"],
                "new_budget": row["new_budget"],
            }
            report.append(record)
            print(record, flush=True)
    (output / "replay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
