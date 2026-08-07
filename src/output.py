"""Persist search results as a single JSON dataset file.

``save_run(path, strategy, n, result)`` appends one run (solved or unsolved).
``load_runs(path)`` loads the full dataset for analysis.
``verify(path)`` checks SHA-256 integrity and Hadamard property of all solved entries.
"""

from __future__ import annotations

import base64
import hashlib
import json
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


SOLUTIONS = Path(__file__).resolve().parent.parent / "data" / "solutions.json"


def _write_entry(
    path: Path, strategy: str, n: int, entry: dict[str, Any], *, dedup_class: bool = False
) -> None:
    existing: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    entries = existing.setdefault(strategy, {}).setdefault(str(n), [])
    # dedup by sha256 (always)
    for prev in entries:
        if prev.get("sha256") == entry["sha256"]:
            return
    # dedup by class hash (for solutions.json — same equivalence class)
    if dedup_class:
        cls = entry.get("class")
        if cls is not None:
            for prev in entries:
                if prev.get("class") == cls:
                    return
    entries.append(entry)
    existing[strategy][str(n)].sort(key=lambda r: r["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def save_run(path: Path, strategy: str, n: int, result: Result) -> None:
    """Append one run to the dataset JSON, solved or unsolved.

    Solved runs (energy==0) are *also* persisted to ``SOLUTIONS`` automatically.
    """
    seqs = result.sequences
    if seqs is None:
        return
    b64 = _seqs_to_b64(seqs)
    sha = hashlib.sha256(seqs.astype(np.int8, copy=False).tobytes()).hexdigest()

    entry: dict[str, Any] = {
        "seed": result.seed,
        "sha256": sha,
        "seqs_b64": b64,
        "solved": result.metrics.energy == 0,
        "energy": result.metrics.energy,
        "solver_e": result.solver_e,
        "elapsed_s": round(result.elapsed, 3),
        "iterations": result.iterations,
    }
    if result.stats:
        entry["stats"] = result.stats.to_dict()
    entry["class"] = Builder.class_hash(seqs)

    _write_entry(path, strategy, n, entry)

    if result.metrics.energy == 0 and path.resolve() != SOLUTIONS.resolve():
        _write_entry(SOLUTIONS, strategy, n, entry, dedup_class=True)


def load_runs(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load all runs from a dataset JSON. Returns ``{strategy: {n: [entries]}}``."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def verify(path: Path) -> int:
    """Verify all solved entries: SHA-256 integrity + pure-Python Hadamard audit."""
    from .generator import Generator
    from .verify import independent_audit

    if not path.exists():
        print(f"{path} not found")
        return 0
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    total = 0
    ok = 0
    for strategy, ns in data.items():
        for n_str, entries in ns.items():
            n = int(n_str)
            solved = [e for e in entries if e.get("solved", e.get("energy", 1) == 0)]
            if not solved:
                continue
            gen = Generator(kind=strategy, n=n)
            for r in solved:
                total += 1
                b64 = r.get("seqs_b64")
                if not b64:
                    print(
                        f"  SKIP {strategy} n={n_str} seed={r.get('seed', '?')}: missing seqs_b64"
                    )
                    continue
                seqs = _seqs_from_b64(b64, (gen._builder.k, n))
                actual_sha = hashlib.sha256(seqs.astype(np.int8, copy=False).tobytes()).hexdigest()
                if actual_sha != r.get("sha256", ""):
                    print(
                        f"  CORRUPT {strategy} n={n_str} seed={r.get('seed', '?')}: sha256 mismatch"
                    )
                    continue
                H = gen._builder.build(seqs)
                try:
                    independent_audit(H.tolist())
                    ok += 1
                except Exception as e:
                    print(f"  FAIL {strategy} n={n_str} seed={r.get('seed', '?')}: {e}")
    print(f"{ok}/{total} valid.")
    return ok
