#!/usr/bin/env python3
"""Exact verifier for a Hadamard matrix submitted as CSV.

The fast check uses NumPy int64 matrix multiplication. This is exact for the
orders relevant here: each dot product is a sum of ``order`` values in {-1, 1}.
The optional audit check recomputes every row-pair dot product in plain Python,
providing a deliberately separate implementation for high-confidence review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

import numpy as np


class InvalidMatrix(ValueError):
    """Raised when a candidate does not satisfy the acceptance contract."""


MAX_CANDIDATE_BYTES = 2_000_000


def read_regular_file(path: Path) -> bytes:
    """Read one bounded regular file through a single no-follow descriptor."""

    flags = os.O_RDONLY | os.O_BINARY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    # Opening a FIFO read-only can block before fstat tells us it is not regular.
    # ponytail: O_NONBLOCK skipped on Windows — makes os.read() return 0 prematurely.
    if os.name != "nt":
        flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InvalidMatrix(f"could not open {path}: {exc}") from exc

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise InvalidMatrix("candidate must be a regular file")
        if before.st_size > MAX_CANDIDATE_BYTES:
            raise InvalidMatrix(
                f"candidate exceeds the {MAX_CANDIDATE_BYTES}-byte safety limit"
            )
        chunks: list[bytes] = []
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, min(
                65_536, MAX_CANDIDATE_BYTES + 1 - bytes_read))
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read > MAX_CANDIDATE_BYTES:
                raise InvalidMatrix(
                    f"candidate exceeds the {MAX_CANDIDATE_BYTES}-byte safety limit"
                )
        after = os.fstat(descriptor)
    except OSError as exc:
        raise InvalidMatrix(f"could not read {path}: {exc}") from exc
    finally:
        os.close(descriptor)

    stable_fields = ("st_dev", "st_ino", "st_size",
                     "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise InvalidMatrix("candidate changed while it was being read")

    raw_bytes = b"".join(chunks)
    if len(raw_bytes) != before.st_size:
        raise InvalidMatrix("candidate changed while it was being read")
    return raw_bytes


def parse_matrix(
    raw_bytes: bytes, expected_order: int
) -> tuple[list[list[int]], np.ndarray]:

    try:
        text = raw_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise InvalidMatrix("candidate must contain ASCII text only") from exc

    # Accept LF and CRLF, with or without one final line ending. Reject every
    # other carriage return/control-line separator explicitly.
    normalized_text = text.replace("\r\n", "\n")
    if "\r" in normalized_text:
        raise InvalidMatrix("candidate may use only LF or CRLF line endings")
    raw_lines = normalized_text.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()
    if any(line == "" for line in raw_lines):
        raise InvalidMatrix("candidate must not contain blank lines")
    raw_rows = [line.split(",") for line in raw_lines]
    if len(raw_rows) != expected_order:
        raise InvalidMatrix(
            f"expected {expected_order} rows, found {len(raw_rows)}"
        )

    rows: list[list[int]] = []
    for row_number, raw_row in enumerate(raw_rows, start=1):
        if len(raw_row) != expected_order:
            raise InvalidMatrix(
                f"row {row_number}: expected {expected_order} entries, "
                f"found {len(raw_row)}"
            )

        row: list[int] = []
        for column_number, token in enumerate(raw_row, start=1):
            if token not in {"-1", "1"}:
                raise InvalidMatrix(
                    f"row {row_number}, column {column_number}: "
                    f"expected -1 or 1, found {token!r}"
                )
            row.append(int(token))
        rows.append(row)

    return rows, np.asarray(rows, dtype=np.int64)


def load_matrix_with_hash(
    path: Path, expected_order: int
) -> tuple[list[list[int]], np.ndarray, str]:
    raw_bytes = read_regular_file(path)
    rows, matrix = parse_matrix(raw_bytes, expected_order)
    return rows, matrix, hashlib.sha256(raw_bytes).hexdigest()


def load_matrix(path: Path, expected_order: int) -> tuple[list[list[int]], np.ndarray]:
    rows, matrix, _raw_sha256 = load_matrix_with_hash(path, expected_order)
    return rows, matrix


def normalized_sha256(rows: list[list[int]]) -> str:
    canonical = "\n".join(",".join(str(value)
                          for value in row) for row in rows)
    return hashlib.sha256((canonical + "\n").encode("ascii")).hexdigest()


def fast_verify(matrix: np.ndarray) -> None:
    order = matrix.shape[0]
    gram = matrix @ matrix.T
    expected = np.eye(order, dtype=np.int64) * order
    mismatch = np.argwhere(gram != expected)
    if mismatch.size:
        i, j = (int(value) for value in mismatch[0])
        raise InvalidMatrix(
            f"orthogonality failure at rows {i + 1} and {j + 1}: "
            f"dot product is {int(gram[i, j])}, expected {int(expected[i, j])}"
        )


def independent_audit(rows: list[list[int]]) -> None:
    order = len(rows)
    for i in range(order):
        if sum(value * value for value in rows[i]) != order:
            raise InvalidMatrix(f"row {i + 1} has incorrect squared length")
        for j in range(i):
            dot_product = sum(a * b for a, b in zip(rows[i], rows[j]))
            if dot_product != 0:
                raise InvalidMatrix(
                    f"independent audit failed for rows {j + 1} and {i + 1}: "
                    f"dot product is {dot_product}"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify an exact ±1 Hadamard matrix stored as CSV."
    )
    parser.add_argument("candidate", type=Path, help="candidate CSV file")
    parser.add_argument(
        "--order",
        type=int,
        default=668,
        help="required matrix order (default: 668)",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="also run an independent pure-Python row-pair audit",
    )
    parser.add_argument(
        "--require-filename",
        metavar="NAME",
        help="reject the candidate unless its basename exactly matches NAME",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    try:
        if args.order <= 0:
            raise InvalidMatrix("order must be positive")
        if args.require_filename and args.candidate.name != args.require_filename:
            raise InvalidMatrix(
                f"candidate filename must be exactly {args.require_filename!r}"
            )
        rows, matrix, raw_sha256 = load_matrix_with_hash(
            args.candidate, args.order)
        fast_verify(matrix)
        if args.audit:
            independent_audit(rows)
    except InvalidMatrix as exc:
        print(
            json.dumps(
                {
                    "valid": False,
                    "order": args.order,
                    "error": str(exc),
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                },
                indent=2,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "valid": True,
                "order": args.order,
                "raw_sha256": raw_sha256,
                "canonical_sha256": normalized_sha256(rows),
                "independent_audit": args.audit,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
