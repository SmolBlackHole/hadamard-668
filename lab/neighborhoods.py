"""Read-only exact repair and specified-group orbit diagnostics.

Run with ``python -m lab.neighborhoods``. Outputs are local research
artifacts, never updates to stored identities or claims of Hadamard equivalence.

For n >= 3, work in R = Z[z]/(z**n+1), with star sending z to z**-1.
Substitution by a unit k modulo 2n is a signed coefficient permutation
automorphism commuting with star. It therefore sends sum(a_i * a_i.star())
to its signed lag permutation and preserves residual squared norm. Independent
negashifts and reversals preserve each row's autocorrelation. Substitution
normalizes these actions (shift t becomes shift kt) and commutes with row
permutations. Enumerating every unit then taking the quick representative is
therefore exact canonicalization under this specified generated group. This
does not enumerate general Hadamard row/column equivalence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sqlite3
from collections.abc import Sequence
from itertools import combinations
from math import gcd
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt

from src.canonical import canonical_sequences, orbit_hash
from src.models import Int8Array
from src.tracker import Tracker
from src.verify import verify_candidate


def residual_rows(rows: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
    """Compute unreduced negaperiodic residuals for a batch of rows."""
    values = rows.astype(np.int64)
    n = values.shape[-1]
    return np.stack(
        [
            np.sum(values[..., :-lag] * values[..., lag:], axis=-1)
            - np.sum(values[..., -lag:] * values[..., :lag], axis=-1)
            for lag in range(1, (n - 1) // 2 + 1)
        ],
        axis=-1,
    )


def decimate(sequences: Int8Array, k: int) -> Int8Array:
    """Apply common substitution z -> z**k in Z[z]/(z**n+1)."""
    n = sequences.shape[1]
    if gcd(k, 2 * n) != 1:
        raise ValueError("decimation must be a unit modulo 2n")
    exponents: npt.NDArray[np.int64] = np.arange(n, dtype=np.int64) * k
    result = np.empty_like(sequences)
    result[:, exponents % n] = sequences * np.where((exponents // n) % 2, -1, 1)
    return result


def extended_hash(sequences: Int8Array) -> str:
    """Canonicalize quick actions plus common unit decimation, not all equivalence."""
    n = sequences.shape[1]
    representative = min(
        canonical_sequences(decimate(sequences, k)).tobytes()
        for k in range(1, 2 * n)
        if gcd(k, 2 * n) == 1
    )
    return hashlib.sha256(n.to_bytes(8, "big") + representative).hexdigest()


def repair_witnesses(sequences: Int8Array) -> dict[str, list[list[int]] | None]:
    """Exhaust singles/pairs/triples and one-per-row quadruples.

    Results give one witness per successful family. All negative results exhaust
    that family. Same-row pair effects are recomputed, never assumed additive.
    """
    n = sequences.shape[1]
    row_residual = residual_rows(sequences)
    target = -row_residual.sum(axis=0)
    variants = np.repeat(sequences[:, None, :], n, axis=1)
    variants[:, np.arange(n), np.arange(n)] *= -1
    delta = residual_rows(variants) - row_residual[:, None, :]
    result: dict[str, list[list[int]] | None] = dict.fromkeys(
        ["single", "pair", "triple_distinct_rows", "triple_all", "quadruple_one_per_row"]
    )

    def save(family: str, moves: list[list[int]]) -> None:
        candidate = sequences.copy()
        for row, column in moves:
            candidate[row, column] *= -1
        verify_candidate(candidate)
        result[family] = moves

    for row in range(4):
        hits = np.flatnonzero(np.all(delta[row] == target, axis=1))
        if len(hits):
            save("single", [[row, int(hits[0])]])
        columns = np.asarray(list(combinations(range(n), 2)))
        pairs = np.repeat(sequences[row][None, :], len(columns), axis=0)
        pairs[np.arange(len(columns))[:, None], columns] *= -1
        effects = residual_rows(pairs) - row_residual[row]
        hits = np.flatnonzero(np.all(effects == target, axis=1))
        if len(hits):
            save("pair", [[row, int(column)] for column in columns[hits[0]]])

    pair_maps: dict[tuple[int, int], dict[bytes, tuple[int, int]]] = {}
    for a, b in combinations(range(4), 2):
        sums = delta[a, :, None, :] + delta[b, None, :, :]
        pair_maps[a, b] = {sums[i, j].tobytes(): (i, j) for i in range(n) for j in range(n)}
        hit = pair_maps[a, b].get(target.tobytes())
        if hit is not None:
            save("pair", [[a, hit[0]], [b, hit[1]]])
    for a, b, c in combinations(range(4), 3):
        for column in range(n):
            hit = pair_maps[a, b].get((target - delta[c, column]).tobytes())
            if hit is not None:
                save("triple_distinct_rows", [[a, hit[0]], [b, hit[1]], [c, column]])
                break
    result["triple_all"] = result["triple_distinct_rows"]
    if result["triple_all"] is None:
        witness = repeated_row_triple(sequences)
        if witness is not None:
            save("triple_all", witness)
    for i in range(n):
        for j in range(n):
            hit = pair_maps[0, 1].get((target - delta[2, i] - delta[3, j]).tobytes())
            if hit is not None:
                save("quadruple_one_per_row", [[0, hit[0]], [1, hit[1]], [2, i], [3, j]])
                return result
    return result


def repeated_row_triple(sequences: Int8Array) -> list[list[int]] | None:
    """Exhaust triples containing a same-row pair with exact updated caches."""
    n = sequences.shape[1]
    state = sequences.copy()
    tracker = Tracker()
    tracker.build(state)
    for row in range(4):
        for first in range(n - 1):
            tracker.accept(state, row, first)
            for second in range(first + 1, n):
                tracker.accept(state, row, second)
                scores = tracker.flip_qs()
                scores[row * n + first] = 1
                scores[row * n + second] = 1
                hits = np.flatnonzero(scores == 0)
                if len(hits):
                    other_row, column = divmod(int(hits[0]), n)
                    return [[row, first], [row, second], [other_row, column]]
                tracker.accept(state, row, second)
            tracker.accept(state, row, first)
    return None


def main(argv: Sequence[str] | None = None) -> None:
    """Analyze the recorded sweep endpoints and all currently valid solutions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/hadamard.db"))
    parser.add_argument("--output", type=Path, default=Path("runs/research/geometry.json"))
    parser.add_argument(
        "--first-run-id",
        type=int,
        default=1,
        help="First inclusive run ID",
    )
    parser.add_argument(
        "--last-run-id",
        type=int,
        default=2**63 - 1,
        help="Last inclusive run ID",
    )
    args = parser.parse_args(argv)
    if args.first_run_id < 1 or args.last_run_id < args.first_run_id:
        parser.error("run ID range must satisfy 1 <= first-run-id <= last-run-id")
    started = perf_counter()
    connection = sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    with connection:
        endpoints = connection.execute(
            "SELECT id,n,seed,seqs_b64 FROM runs WHERE id BETWEEN ? AND ? "
            "AND strategy='gs4' AND valid=1 AND energy=64*n ORDER BY id",
            (args.first_run_id, args.last_run_id),
        ).fetchall()
        solutions = connection.execute(
            "SELECT id,n,seqs_b64 FROM solutions WHERE valid=1 ORDER BY id"
        ).fetchall()
    connection.close()
    input_payload = {
        "endpoints": [dict(row) for row in endpoints],
        "solutions": [dict(row) for row in solutions],
    }
    input_sha256 = hashlib.sha256(
        json.dumps(input_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    def decode(row: sqlite3.Row) -> Int8Array:
        return np.frombuffer(base64.b64decode(row["seqs_b64"]), dtype=np.int8).reshape(4, row["n"])

    repairs: list[dict[str, Any]] = []
    for row in endpoints:
        state = decode(row)
        residual = residual_rows(state).sum(axis=0)
        if int(residual @ residual) != 16:
            raise ValueError(f"stored Q=1 does not match endpoint {row['id']}")
        repairs.append({"id": row["id"], "n": row["n"], "witnesses": repair_witnesses(state)})
    classes: list[dict[str, Any]] = []
    for row in solutions:
        state = decode(row)
        verify_candidate(state)
        classes.append(
            {
                "id": row["id"],
                "n": row["n"],
                "quick": orbit_hash(state),
                "extended": extended_hash(state),
            }
        )
    payload = {
        "scope": "selected inclusive run ID range, gs4 Q=1 endpoints; all valid solution rows",
        "first_run_id": args.first_run_id,
        "last_run_id": args.last_run_id,
        "group": "independent negashifts/reversals, row permutations, common unit decimation",
        "limitation": "not full Hadamard equivalence; no unrestricted quadruples",
        "database": str(args.database),
        "input_sha256": input_sha256,
        "input_hash_format": "UTF-8 sorted-key compact JSON of ordered selected endpoint and solution records",
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repairs": repairs,
        "solutions": classes,
        "elapsed_s": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "endpoints": len(repairs),
                "solutions": len(classes),
                "elapsed_s": payload["elapsed_s"],
                "repairable": {
                    family: sum(item["witnesses"][family] is not None for item in repairs)
                    for family in (
                        "single",
                        "pair",
                        "triple_distinct_rows",
                        "triple_all",
                        "quadruple_one_per_row",
                    )
                },
            }
        )
    )


if __name__ == "__main__":
    main()
