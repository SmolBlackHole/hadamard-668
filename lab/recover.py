"""Measure the current solver's recovery from controlled solution perturbations."""

from __future__ import annotations

import argparse
import base64
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from lab.diagnose import pair_diagnosis
from lab.provenance import capture_source
from src.models import Int8Array
from src.solver import search
from src.tracker import Tracker
from src.verify import verify_candidate


def decode(encoded: str, n: int) -> Int8Array:
    """Decode an owned sequence array from a database record."""
    return np.frombuffer(base64.b64decode(encoded), dtype=np.int8).reshape(4, n).copy()


def geometry(state: Int8Array) -> dict[str, int]:
    """Measure singles and use the shared exact pair diagnostic."""
    tracker = Tracker()
    tracker.build(state)
    q = tracker.energy() // (64 * state.shape[1])
    singles = tracker.flip_qs()
    return {
        "q": q,
        "best_single": int(singles.min()),
        "best_pair": pair_diagnosis(state)["best_pair_q"],
        "improving_singles": int(np.sum(singles < q)),
        "neutral_singles": int(np.sum(singles == q)),
    }


def descend(state: Int8Array) -> Int8Array:
    """Apply at most 64 steepest improving flips without modifying the input."""
    state = state.copy()
    tracker = Tracker()
    tracker.build(state)
    for _ in range(64):
        values = tracker.flip_qs()
        index = int(values.argmin())
        if values[index] >= tracker.energy() // (64 * state.shape[1]):
            break
        state.flat[index] *= -1
        tracker.build(state)
    return state


def main(argv: Sequence[str] | None = None) -> None:
    """Run explicit parent IDs through the production solver, preserving inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/hadamard.db"))
    parser.add_argument("--parent-ids", type=int, nargs="+", required=True)
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--candidate-budget", type=int, default=250_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.repeats, args.candidate_budget, *args.radii) < 1 or args.seed < 0:
        parser.error("radii, repeats and budget must be positive; seed must be nonnegative")
    if not args.database.is_file():
        parser.error("database does not exist")
    parents: list[dict[str, Any]] = []
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for identifier in dict.fromkeys(args.parent_ids):
            row = connection.execute("SELECT * FROM runs WHERE id=?", (identifier,)).fetchone()
            if row is None or row["valid"] != 1 or not row["solved"]:
                parser.error(f"parent {identifier} must be a validated solved run")
            if max(args.radii) > 4 * row["n"]:
                parser.error(f"radius exceeds parent {identifier}'s state size")
            verify_candidate(decode(row["seqs_b64"], row["n"]))
            parents.append(dict(row))
    try:
        provenance = capture_source(args.output)
    except FileExistsError as error:
        parser.error(str(error))
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    for parent in parents:
        for radius in dict.fromkeys(args.radii):
            for repeat in range(args.repeats):
                state = decode(parent["seqs_b64"], parent["n"])
                flips: npt.NDArray[np.int64] = rng.choice(state.size, int(radius), replace=False)
                state.flat[flips] *= -1
                tracker = Tracker()
                tracker.build(state)
                start_q = tracker.energy() // (64 * parent["n"])
                result = search(
                    state,
                    Tracker(),
                    np.random.default_rng(args.seed + repeat),
                    candidate_budget=args.candidate_budget,
                )
                if result.solved:
                    verify_candidate(result.sequences)
                records.append(
                    {
                        "parent_id": parent["id"],
                        "n": parent["n"],
                        "radius": radius,
                        "flips": flips.tolist(),
                        "seed": args.seed + repeat,
                        "start_q": start_q,
                        "end_q": result.energy // (64 * parent["n"]),
                        "candidate_evals": result.candidate_evals,
                    }
                )
        print(f"parent {parent['id']}: completed", flush=True)
    report = {
        "provenance": provenance,
        "database": str(args.database.resolve()),
        "budget": args.candidate_budget,
        "perturbation_seed": args.seed,
        "parents": parents,
        "runs": records,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure = Figure(figsize=(8, 5), layout="constrained")
    axis = figure.subplots()
    for parent in parents:
        radii = sorted(set(args.radii))
        rates = [
            sum(
                r["end_q"] == 0
                for r in records
                if r["parent_id"] == parent["id"] and r["radius"] == radius
            )
            / args.repeats
            for radius in radii
        ]
        axis.plot(radii, rates, "o-", label=f"n={parent['n']}, parent={parent['id']}")  # pyright: ignore[reportUnknownMemberType]
    axis.set(
        title="Recovery by parent (dependent perturbations)",
        xlabel="Flipped bits",
        ylabel="Solved fraction",
        ylim=(0, 1.05),
    )
    axis.legend()  # pyright: ignore[reportUnknownMemberType]
    for extension in ("png", "svg"):
        figure.savefig(args.output.with_suffix("." + extension), dpi=160)  # pyright: ignore[reportUnknownMemberType]
    print(f"Results and plots: {args.output}")


if __name__ == "__main__":
    main()
