"""Benchmark: random vs cyclic starts (SQLite-backed, worker-safe)."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

PROJ = Path(__file__).resolve().parent.parent


class SweepEntry(TypedDict):
    """Persisted outcome fields used by the start-state benchmark."""

    seed: int
    solved: bool
    energy: int
    elapsed_s: float


def run_sweep(
    n: int,
    seeds: int,
    candidate_budget: int,
    extra_args: Sequence[str],
    label: str,
) -> list[SweepEntry]:
    """Run one isolated start-state sweep and remove its temporary database.

    Args:
        n: GS4 sequence length.
        seeds: Number of consecutive seeds to run.
        candidate_budget: Candidate evaluations allowed per seed.
        extra_args: Additional arguments forwarded to ``run.py``.
        label: File-safe label for terminal output and temporary storage.

    Returns:
        Outcomes loaded from the temporary result database.

    Note:
        ``run_sweep()`` starts a full solver benchmark. Importing the module
        does not execute it; invoke the script explicitly when the cost is
        intended.
    """
    out = PROJ / f"data/_bench_{n}_{label}.db"
    try:
        cmd: list[str] = [
            sys.executable,
            str(PROJ / "run.py"),
            "--sweep",
            "gs4",
            str(n),
            "--seeds",
            str(seeds),
            "--candidate-budget",
            str(candidate_budget),
            "--workers",
            "8",
            "--output",
            str(out),
            "--no-verify",
            *extra_args,
        ]
        subprocess.run(cmd, check=True, cwd=PROJ)
        con = sqlite3.connect(out)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT seed, solved, energy, elapsed_s FROM runs "
            "WHERE strategy='gs4' AND n=? ORDER BY seed",
            (n,),
        ).fetchall()
        con.close()
        entries = [
            SweepEntry(
                seed=int(row["seed"]),
                solved=bool(row["solved"]),
                energy=int(row["energy"]),
                elapsed_s=float(row["elapsed_s"]),
            )
            for row in rows
        ]
        solved = sum(entry["solved"] for entry in entries)
        best_e = min((entry["energy"] for entry in entries), default=999999)
        best_q = best_e // (64 * n)
        t_total = sum(entry["elapsed_s"] for entry in entries)
        q_vals = [entry["energy"] // (64 * n) for entry in entries]
        print(
            f"  {label}: n={n} {solved}/{len(entries)} solved "
            f"best Q={best_q} Q-dist={sorted(q_vals)[:5]}... {t_total:.1f}s"
        )
        return entries
    finally:
        Path(out).unlink(missing_ok=True)
        Path(f"{out}-wal").unlink(missing_ok=True)
        Path(f"{out}-shm").unlink(missing_ok=True)


if __name__ == "__main__":
    for n, seeds, candidate_budget in [(44, 30, 100_000_000)]:
        print(f"\n=== n={n} ===")
        run_sweep(n, seeds, candidate_budget, ["--no-targeted-escape"], "random")
        run_sweep(
            n,
            seeds,
            candidate_budget,
            ["--start-kind", "cyclic", "--no-targeted-escape"],
            "cyclic",
        )
