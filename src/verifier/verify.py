"""Independent reference verification for Hadamard constructions.

Pure-Python implementations that deliberately avoid the solver's
NumPy/CuPy code paths. Used to confirm exact solutions.
"""

from __future__ import annotations

from typing import Sequence


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
            dot = sum(a * b for a, b in zip(rows[i], rows[j]))
            if dot != 0:
                raise InvalidMatrix(
                    f"independent audit failed for rows {j + 1} and {i + 1}: dot product is {dot}"
                )


def verify_goethals_seidel_sequences(
    sequences: Sequence[Sequence[int]],
) -> None:
    values = tuple(_sign_sequence(s, n) for n, s in zip("ABCD", sequences))
    if len(values) != 4 or len({len(s) for s in values}) != 1:
        raise InvalidMatrix("G-S requires four equally long sign sequences")
    size = len(values[0])
    for shift in range(1, size):
        corr = sum(seq[i] * seq[(i + shift) % size] for seq in values for i in range(size))
        if corr:
            raise InvalidMatrix(f"G-S autocorrelation at shift {shift} is {corr}")


def verify_turyn_sequences(
    sequences: Sequence[Sequence[int]],
) -> None:
    values = tuple(_sign_sequence(s, n) for n, s in zip("XYZW", sequences))
    if len(values) != 4:
        raise InvalidMatrix("Turyn type requires four sign sequences")
    size = len(values[0])
    if [len(s) for s in values] != [size, size, size, size - 1]:
        raise InvalidMatrix("Turyn sequences must have lengths (n, n, n, n-1)")
    for shift in range(1, size):
        corr = sum(
            w * sum(s[i] * s[i + shift] for i in range(len(s) - shift))
            for s, w in zip(values, (1, 1, 2, 2))
        )
        if corr:
            raise InvalidMatrix(f"Turyn autocorrelation at shift {shift} is {corr}")


# ── Reference builders ────────────────────────────────────────────────────────


def _circulant(values: list[int]) -> list[list[int]]:
    n = len(values)
    return [[values[(c - r) % n] for c in range(n)] for r in range(n)]


def _transpose(m: list[list[int]]) -> list[list[int]]:
    return [list(col) for col in zip(*m)]


def _reverse_columns(m: list[list[int]]) -> list[list[int]]:
    return [r[::-1] for r in m]


def _turyn_sign_sequences(
    sequences: Sequence[Sequence[int]],
) -> tuple[list[int], list[int], list[int], list[int]]:
    values = tuple(_sign_sequence(s, n) for n, s in zip("XYZW", sequences))
    if len(values) != 4:
        raise InvalidMatrix("Turyn type requires four sign sequences")
    x, y, z, w = values
    n = len(x)
    if [len(s) for s in (x, y, z, w)] != [n, n, n, n - 1]:
        raise InvalidMatrix("Turyn sequences must have lengths (n, n, n, n-1)")
    a, b, c, d = z + w, z + [-v for v in w], x, y
    short, long = len(c), len(a)
    t1 = [(lx + rx) // 2 for lx, rx in zip(a, b)] + [0] * short
    t2 = [(lx - rx) // 2 for lx, rx in zip(a, b)] + [0] * short
    t3 = [0] * long + [(lx + rx) // 2 for lx, rx in zip(c, d)]
    t4 = [0] * long + [(lx - rx) // 2 for lx, rx in zip(c, d)]
    if any(sum(abs(s[i]) for s in (t1, t2, t3, t4)) != 1 for i in range(len(t1))):
        raise InvalidMatrix("Turyn conversion did not produce disjoint T-sequences")
    return (
        [a + b + c + d for a, b, c, d in zip(t1, t2, t3, t4)],
        [-a + b + c - d for a, b, c, d in zip(t1, t2, t3, t4)],
        [-a - b + c + d for a, b, c, d in zip(t1, t2, t3, t4)],
        [-a + b - c + d for a, b, c, d in zip(t1, t2, t3, t4)],
    )


def build_goethals_seidel_reference(
    sequences: Sequence[Sequence[int]],
) -> list[list[int]]:
    values = tuple(_sign_sequence(s, n) for n, s in zip("ABCD", sequences))
    if len(values) != 4 or len({len(s) for s in values}) != 1:
        raise InvalidMatrix("G-S requires four equally long sign sequences")
    a, b, c, d = (_circulant(s) for s in values)
    br, cr, dr = (_reverse_columns(m) for m in (b, c, d))
    btr, ctr, dtr = (_reverse_columns(_transpose(m)) for m in (b, c, d))
    blocks = ((a, br, cr, dr), (br, a, dtr, ctr), (cr, dtr, a, btr), (dr, ctr, btr, a))
    signs = ((1, 1, 1, 1), (-1, 1, -1, 1), (-1, 1, 1, -1), (-1, -1, 1, 1))
    return [
        [sig * val for block, sig in zip(block_row, sign_row) for val in block[row]]
        for block_row, sign_row in zip(blocks, signs)
        for row in range(len(a))
    ]


def build_turyn_reference(
    sequences: Sequence[Sequence[int]],
) -> list[list[int]]:
    return build_goethals_seidel_reference(_turyn_sign_sequences(sequences))


def verify_turyn_candidate(sequences: Sequence[Sequence[int]]) -> list[list[int]]:
    verify_turyn_sequences(sequences)
    verify_goethals_seidel_sequences(_turyn_sign_sequences(sequences))
    matrix = build_turyn_reference(sequences)
    independent_audit(matrix)
    return matrix
