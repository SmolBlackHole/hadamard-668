"""SQLite-backed persistence for search results (replaces the JSON dataset).

``save_run(path, strategy, n, result)`` stores one run (solved or unsolved);
solved runs (energy==0) are also recorded in the ``solutions`` table.
``load_runs(path)`` loads the full dataset for analysis.
``verify(path)`` checks SHA-256 integrity and Hadamard property of all solved entries.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .builder import Builder
from .generator import Result


def _seqs_to_b64(seqs: np.ndarray) -> str:
    return base64.b64encode(seqs.astype(np.int8, copy=False).tobytes()).decode("ascii")


def _seqs_from_b64(b64: str, shape: tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.int8).reshape(shape)


SOLUTIONS = Path(__file__).resolve().parent.parent / "data" / "hadamard.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    n INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    seqs_b64 TEXT NOT NULL,
    solved INTEGER NOT NULL DEFAULT 0,
    energy INTEGER NOT NULL,
    solver_e INTEGER NOT NULL DEFAULT 0,
    elapsed_s REAL NOT NULL DEFAULT 0.0,
    iterations INTEGER NOT NULL DEFAULT 0,
    class_hash TEXT,
    stats_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS solutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    n INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    seqs_b64 TEXT NOT NULL,
    energy INTEGER NOT NULL DEFAULT 0,
    solver_e INTEGER NOT NULL DEFAULT 0,
    elapsed_s REAL NOT NULL DEFAULT 0.0,
    iterations INTEGER NOT NULL DEFAULT 0,
    class_hash TEXT NOT NULL,
    stats_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_runs_strategy_n ON runs(strategy, n);
CREATE INDEX IF NOT EXISTS idx_runs_sha256 ON runs(sha256);
CREATE INDEX IF NOT EXISTS idx_solutions_class ON solutions(class_hash);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a WAL-mode connection and ensure the schema; caller commits and closes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript(_SCHEMA)
    return con


def save_run(db_path: Path, strategy: str, n: int, result: Result) -> None:
    """Store one run in the database; solved runs are also recorded in ``solutions``."""
    seqs = result.sequences
    if seqs is None:
        return
    b64 = _seqs_to_b64(seqs)
    raw = seqs.astype(np.int8, copy=False).tobytes()
    sha = hashlib.sha256(raw).hexdigest()
    solved = result.metrics.energy == 0
    stats_json = json.dumps(result.stats.to_dict()) if result.stats else None
    class_hash = Builder.class_hash(seqs)

    con = _connect(db_path)
    try:
        con.execute("BEGIN IMMEDIATE")
        dup = con.execute(
            "SELECT 1 FROM runs WHERE strategy = ? AND n = ? AND sha256 = ?",
            (strategy, n, sha),
        ).fetchone()
        if not dup:
            con.execute(
                "INSERT INTO runs (strategy, n, seed, sha256, seqs_b64, solved, energy,"
                " solver_e, elapsed_s, iterations, class_hash, stats_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (strategy, n, result.seed, sha, b64, int(solved), result.metrics.energy,
                 result.solver_e, round(result.elapsed, 3), result.iterations,
                 class_hash, stats_json),
            )
        if solved:
            existing = con.execute(
                "SELECT 1 FROM solutions WHERE class_hash = ?", (class_hash,)
            ).fetchone()
            if not existing:
                con.execute(
                    "INSERT OR IGNORE INTO solutions (strategy, n, seed, sha256, seqs_b64,"
                    " energy, solver_e, elapsed_s, iterations, class_hash, stats_json)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (strategy, n, result.seed, sha, b64, result.metrics.energy,
                     result.solver_e, round(result.elapsed, 3), result.iterations,
                     class_hash, stats_json),
                )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def load_runs(db_path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load all runs as ``{strategy: {n: [entries]}}`` (same shape as the old JSON format)."""
    if not Path(db_path).exists():
        return {}
    con = _connect(db_path)
    try:
        rows = con.execute(
            "SELECT strategy, n, seed, sha256, seqs_b64, solved, energy, solver_e,"
            " elapsed_s, iterations, class_hash, stats_json FROM runs"
            " ORDER BY strategy, n, seed"
        ).fetchall()
    finally:
        con.close()
    data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        entry: dict[str, Any] = {
            "seed": r["seed"],
            "sha256": r["sha256"],
            "seqs_b64": r["seqs_b64"],
            "solved": bool(r["solved"]),
            "energy": r["energy"],
            "solver_e": r["solver_e"],
            "elapsed_s": r["elapsed_s"],
            "iterations": r["iterations"],
            "class": r["class_hash"],
        }
        if r["stats_json"] is not None:
            entry["stats"] = json.loads(r["stats_json"])
        data.setdefault(r["strategy"], {}).setdefault(str(r["n"]), []).append(entry)
    return data


def verify(db_path: Path) -> int:
    """Verify all solved entries: SHA-256 integrity + pure-Python Hadamard audit."""
    from .generator import Generator
    from .verify import independent_audit

    if not Path(db_path).exists():
        print(f"{db_path} not found")
        return 0
    con = _connect(db_path)
    try:
        rows = con.execute(
            "SELECT strategy, n, seed, sha256, seqs_b64 FROM runs WHERE solved = 1"
            " ORDER BY strategy, n"
        ).fetchall()
    finally:
        con.close()
    total = 0
    ok = 0
    for r in rows:
        strategy = r["strategy"]
        n = r["n"]
        gen = Generator(kind=strategy, n=n)
        total += 1
        seqs = _seqs_from_b64(r["seqs_b64"], (gen._builder.k, n))
        actual_sha = hashlib.sha256(seqs.astype(np.int8, copy=False).tobytes()).hexdigest()
        if actual_sha != r["sha256"]:
            print(f"  CORRUPT {strategy} n={n} seed={r['seed']}: sha256 mismatch")
            continue
        H = gen._builder.build(seqs)
        try:
            independent_audit(H.tolist())
            ok += 1
        except Exception as e:
            print(f"  FAIL {strategy} n={n} seed={r['seed']}: {e}")
    print(f"{ok}/{total} valid.")
    return ok
