"""Independent reference verification for Hadamard constructions.

Pure-Python, deliberately avoids the solver's NumPy/CuPy code paths.
"""

from __future__ import annotations

from collections.abc import Sequence


class InvalidMatrix(ValueError):
    """Raised when a candidate does not satisfy the acceptance contract."""


def _sign_sequence(values: Sequence[int], name: str) -> list[int]:
    seq = list(values)
    if not seq or any(v not in (-1, 1) for v in seq):
        raise InvalidMatrix(f"{name} must be a non-empty sign sequence")
    return seq


def independent_audit(rows: list[list[int]]) -> None:
    """Verify rows form a Hadamard matrix: pure-Python, no NumPy."""
    order = len(rows)
    for i in range(order):
        if any(v not in (-1, 1) for v in rows[i]):
            raise InvalidMatrix(f"row {i + 1} contains a value other than -1 or 1")
        if sum(v * v for v in rows[i]) != order:
            raise InvalidMatrix(f"row {i + 1} has incorrect squared length")
        for j in range(i):
            dot = sum(a * b for a, b in zip(rows[i][:], rows[j][:]))
            if dot != 0:
                raise InvalidMatrix(
                    f"independent audit failed for rows {j + 1} and {i + 1}: dot product is {dot}"
                )
