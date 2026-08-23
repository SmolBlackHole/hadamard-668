"""SQLite-backed persistence for search results and verified solutions.

``save_runs(path, results)`` stores a batch of runs in one transaction;
``save_run(path, result)`` is the single-run convenience wrapper.
Verified zero-energy runs are also recorded in the ``solutions`` table.
``load_runs(path)`` loads the full dataset for analysis.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import subprocess
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import CANONICALIZER, identify_state
from .models import MigrationReport, RunResult


def _seqs_to_b64(seqs: np.ndarray) -> str:
    return base64.b64encode(seqs.astype(np.int8, copy=False).tobytes()).decode("ascii")


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
    candidate_evals INTEGER,
    candidate_budget INTEGER,
    construction TEXT,
    solver_config_json TEXT,
    code_revision TEXT,
    class_hash TEXT,
    validation_hash TEXT,
    orbit_hash TEXT,
    canonical_b64 TEXT,
    canonicalizer TEXT,
    valid INTEGER,
    validation_error TEXT,
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
    candidate_evals INTEGER,
    candidate_budget INTEGER,
    construction TEXT,
    solver_config_json TEXT,
    code_revision TEXT,
    class_hash TEXT NOT NULL,
    validation_hash TEXT,
    orbit_hash TEXT,
    canonical_b64 TEXT,
    canonicalizer TEXT,
    valid INTEGER,
    validation_error TEXT,
    stats_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_runs_strategy_n ON runs(strategy, n);
CREATE INDEX IF NOT EXISTS idx_runs_sha256 ON runs(sha256);
CREATE INDEX IF NOT EXISTS idx_solutions_class ON solutions(class_hash);
CREATE TABLE IF NOT EXISTS solution_features (
    solution_id INTEGER NOT NULL REFERENCES solutions(id),
    feature_version TEXT NOT NULL,
    features_json TEXT NOT NULL,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (solution_id, feature_version)
);
"""

_IDENTITY_COLUMNS = {
    "validation_hash": "TEXT",
    "orbit_hash": "TEXT",
    "canonical_b64": "TEXT",
    "canonicalizer": "TEXT",
    "valid": "INTEGER",
    "validation_error": "TEXT",
    "candidate_evals": "INTEGER",
    "candidate_budget": "INTEGER",
    "construction": "TEXT",
    "solver_config_json": "TEXT",
    "code_revision": "TEXT",
}


def _ensure_identity_columns(connection: sqlite3.Connection) -> None:
    """Add identity and provenance columns missing from a legacy database."""
    for table in ("runs", "solutions"):
        present = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        for name, sql_type in _IDENTITY_COLUMNS.items():
            if name not in present:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_runs_orbit ON runs(orbit_hash)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_solutions_orbit ON solutions(orbit_hash)")


def _connect(db_path: Path, *, initialize: bool) -> sqlite3.Connection:
    """Open an initialized writer or a genuinely read-only SQLite connection.

    Writer setup creates parent directories, enables WAL, applies the schema,
    and extends legacy tables. Read-only setup never creates a missing file.
    """
    if not initialize:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript(_SCHEMA)
    _ensure_identity_columns(con)
    return con


@cache
def _code_revision() -> str | None:
    """Return the cached Git revision, marked dirty when the worktree differs."""
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    if revision.returncode:
        return None
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    suffix = "-dirty" if status.returncode == 0 and status.stdout else ""
    return revision.stdout.strip() + suffix


def save_runs(
    db_path: Path,
    results: Sequence[RunResult],
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Store a batch atomically and index verified solved runs as solutions.

    Every result is appended to ``runs``. Results whose :attr:`RunResult.solved`
    property is true are inserted into ``solutions`` and deduplicated by their
    exact payload hash.

    Args:
        db_path: SQLite file to create or update.
        results: Completed runs to write in one transaction.
        progress: Optional callback receiving saved and total row counts.

    Raises:
        ValueError: If a sequence payload fails identity validation.
        sqlite3.Error: If the transaction cannot be completed.

    Note:
        Intermediate progress callbacks occur while the transaction is open;
        the final ``(total, total)`` callback occurs after commit. An empty
        input performs no filesystem or callback work.
    """
    if not results:
        return

    revision = _code_revision()
    total = len(results)
    con = _connect(db_path, initialize=True)
    try:
        con.execute("BEGIN IMMEDIATE")
        for saved, result in enumerate(results, 1):
            seqs = result.sequences
            b64 = _seqs_to_b64(seqs)
            identity = identify_state(seqs)
            solved = result.solved
            stats_json = json.dumps(result.stats.to_dict())

            con.execute(
                "INSERT INTO runs (strategy, n, seed, sha256, seqs_b64, solved, energy,"
                " solver_e, elapsed_s, iterations, candidate_evals, candidate_budget,"
                " construction, solver_config_json, code_revision, class_hash,"
                " validation_hash, orbit_hash,"
                " canonical_b64, canonicalizer, valid, validation_error, stats_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.strategy,
                    result.n,
                    result.seed,
                    identity.validation_hash,
                    b64,
                    int(solved),
                    result.energy,
                    result.energy,
                    round(result.elapsed_seconds, 3),
                    0,
                    result.candidate_evals,
                    result.candidate_budget,
                    result.construction,
                    json.dumps(result.solver_config, sort_keys=True),
                    revision,
                    identity.orbit_hash,
                    identity.validation_hash,
                    identity.orbit_hash,
                    identity.canonical_b64,
                    identity.canonicalizer,
                    1,
                    None,
                    stats_json,
                ),
            )
            if solved:
                con.execute(
                    "INSERT OR IGNORE INTO solutions (strategy, n, seed, sha256, seqs_b64,"
                    " energy, solver_e, elapsed_s, iterations, candidate_evals, candidate_budget,"
                    " construction, solver_config_json, code_revision, class_hash,"
                    " validation_hash, orbit_hash, canonical_b64, canonicalizer, valid,"
                    " validation_error, stats_json)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        result.strategy,
                        result.n,
                        result.seed,
                        identity.validation_hash,
                        b64,
                        result.energy,
                        result.energy,
                        round(result.elapsed_seconds, 3),
                        0,
                        result.candidate_evals,
                        result.candidate_budget,
                        result.construction,
                        json.dumps(result.solver_config, sort_keys=True),
                        revision,
                        identity.orbit_hash,
                        identity.validation_hash,
                        identity.orbit_hash,
                        identity.canonical_b64,
                        identity.canonicalizer,
                        1,
                        None,
                        stats_json,
                    ),
                )
            if progress is not None and saved < total:
                progress(saved, total)
        con.execute("COMMIT")
        if progress is not None:
            progress(total, total)
    except BaseException:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def save_run(db_path: Path, result: RunResult) -> None:
    """Store one completed run using :func:`save_runs`.

    Args:
        db_path: SQLite file to create or update.
        result: Completed run to persist.
    """
    save_runs(db_path, (result,))


def load_runs(db_path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load the complete run history into the legacy nested mapping shape.

    Args:
        db_path: Existing SQLite database.

    Returns:
        ``{strategy: {str(n): [run dictionaries]}}`` ordered by strategy,
        length, seed, and insertion ID. A missing database yields an empty
        mapping and is not created.
    """
    if not Path(db_path).exists():
        return {}
    con = _connect(db_path, initialize=False)
    try:
        rows = con.execute(
            "SELECT strategy, n, seed, sha256, seqs_b64, solved, energy, solver_e,"
            " elapsed_s, candidate_evals, candidate_budget, construction, solver_config_json,"
            " code_revision, class_hash, validation_hash,"
            " orbit_hash, canonicalizer, valid, validation_error, stats_json FROM runs"
            " ORDER BY strategy, n, seed, id"
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
            "candidate_evals": r["candidate_evals"],
            "candidate_budget": r["candidate_budget"],
            "construction": r["construction"],
            "solver_config": (
                json.loads(r["solver_config_json"]) if r["solver_config_json"] is not None else None
            ),
            "code_revision": r["code_revision"],
            "class": r["class_hash"],
            "validation_hash": r["validation_hash"],
            "orbit_hash": r["orbit_hash"],
            "canonicalizer": r["canonicalizer"],
            "valid": bool(r["valid"]),
            "validation_error": r["validation_error"],
        }
        if r["stats_json"] is not None:
            entry["stats"] = json.loads(r["stats_json"])
        data.setdefault(r["strategy"], {}).setdefault(str(r["n"]), []).append(entry)
    return data


def load_audit_records(db_path: Path) -> list[dict[str, Any]]:
    """Load raw run and solution rows required by the read-only audit.

    Args:
        db_path: Existing SQLite database.

    Returns:
        Rows from both tables, annotated with ``source_table``. A missing
        database yields an empty list and is not created.
    """
    if not db_path.exists():
        return []
    con = _connect(db_path, initialize=False)
    try:
        rows: list[sqlite3.Row] = []
        for table in ("runs", "solutions"):
            rows.extend(
                con.execute(
                    f"SELECT '{table}' AS source_table, id, strategy, n, seed, sha256,"
                    " seqs_b64, solved, validation_hash, orbit_hash, canonical_b64,"
                    " canonicalizer, valid, validation_error"
                    f" FROM {table} ORDER BY id"
                    if table == "runs"
                    else f"SELECT '{table}' AS source_table, id, strategy, n, seed, sha256,"
                    " seqs_b64, 1 AS solved, validation_hash, orbit_hash, canonical_b64,"
                    " canonicalizer, valid, validation_error"
                    f" FROM {table} ORDER BY id"
                ).fetchall()
            )
    finally:
        con.close()
    return [dict(row) for row in rows]


def migrate_database(db_path: Path) -> MigrationReport:
    """Backfill versioned identities and classify invalid legacy records.

    Both ``runs`` and ``solutions`` are processed in one write transaction.
    Invalid rows remain in place with ``valid = 0`` and their validation error;
    no experimental history is deleted.

    Args:
        db_path: SQLite database to create, extend, or migrate.

    Returns:
        Counts of checked, valid, and quarantined rows.

    Raises:
        sqlite3.Error: If schema setup or the migration transaction fails.
    """
    from .verify import validate_database_record

    con = _connect(db_path, initialize=True)
    checked = 0
    valid = 0
    quarantined = 0
    try:
        con.execute("BEGIN IMMEDIATE")
        for table in ("runs", "solutions"):
            rows = con.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
            for row in rows:
                checked += 1
                record = dict(row)
                record["source_table"] = table
                record["solved"] = bool(record.get("solved", True))
                error, identity = validate_database_record(record, check_stored_identity=False)
                is_valid = error is None
                valid += is_valid
                quarantined += not is_valid
                con.execute(
                    f"UPDATE {table} SET validation_hash=?, orbit_hash=?, canonical_b64=?,"
                    " canonicalizer=?, class_hash=?, valid=?, validation_error=? WHERE id=?",
                    (
                        identity.validation_hash if identity is not None else record["sha256"],
                        identity.orbit_hash if identity is not None else None,
                        identity.canonical_b64 if identity is not None else None,
                        CANONICALIZER if identity is not None else None,
                        identity.orbit_hash if identity is not None else record["class_hash"],
                        int(is_valid),
                        error,
                        record["id"],
                    ),
                )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    return MigrationReport(checked, valid, quarantined)


def load_valid_solutions(db_path: Path) -> list[dict[str, Any]]:
    """Load solution rows currently marked valid.

    Args:
        db_path: Existing SQLite database.

    Returns:
        Solution dictionaries ordered by length and insertion ID. Each includes
        a writable copy of the decoded ``(4, n)`` ``int8`` array under
        ``sequences``. A missing database yields an empty list.
    """
    if not db_path.exists():
        return []
    con = _connect(db_path, initialize=False)
    try:
        rows = con.execute(
            "SELECT id, strategy, n, seed, validation_hash, orbit_hash, seqs_b64"
            " FROM solutions WHERE valid=1 ORDER BY n, id"
        ).fetchall()
    finally:
        con.close()
    return [
        {
            **dict(row),
            "sequences": np.frombuffer(base64.b64decode(row["seqs_b64"]), dtype=np.int8)
            .reshape(4, row["n"])
            .copy(),
        }
        for row in rows
    ]


def store_solution_features(
    db_path: Path, feature_version: str, features: list[tuple[int, dict[str, Any]]]
) -> None:
    """Upsert versioned structural features for persisted solutions.

    Args:
        db_path: SQLite database to create or update.
        feature_version: Stable identifier for the feature schema.
        features: Pairs of solution IDs and JSON-compatible feature mappings.

    Raises:
        sqlite3.Error: If the transactional upsert cannot be completed.
        TypeError: If a feature mapping is not JSON serializable.
    """
    con = _connect(db_path, initialize=True)
    try:
        con.execute("BEGIN IMMEDIATE")
        con.executemany(
            "INSERT INTO solution_features (solution_id, feature_version, features_json)"
            " VALUES (?, ?, ?) ON CONFLICT(solution_id, feature_version) DO UPDATE SET"
            " features_json=excluded.features_json, computed_at=datetime('now')",
            (
                (solution_id, feature_version, json.dumps(values))
                for solution_id, values in features
            ),
        )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()
