"""Explore alternation lifts and Paley/Turyn folds with canonical constructions.

The experiments use the production verifier for claimed solutions. Numerical
identity checks and round trips are controls, not proofs or search benchmarks.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Literal

import numpy as np

from src.constructions import double_gs4, paley_ng_sequences, supports_paley_ng
from src.models import Int8Array
from src.tracker import Tracker
from src.verify import verify_candidate

FoldSource = Literal["paley", "turyn", "alt"]


def alt(sequence: Int8Array) -> Int8Array:
    """Return a copy with alternating signs, starting with a positive sign."""
    result = sequence.copy()
    result[1::2] *= -1
    return result


def alt_double(sequences: Int8Array) -> Int8Array:
    """Interleave row pairs and alternate them, mapping Q to 4Q at twice n.

    Input is a binary array of shape (4, m). The output is (a, alt(a), b,
    alt(b)), with a from rows 0/1 and b from rows 2/3. Zero residual is preserved.
    """
    if sequences.ndim != 2 or sequences.shape[0] != 4 or sequences.shape[1] == 0:
        raise ValueError("alt_double expects shape (4, m), m > 0")
    m = sequences.shape[1]
    a = np.empty(2 * m, dtype=np.int8)
    b = np.empty(2 * m, dtype=np.int8)
    a[0::2], a[1::2] = sequences[0], sequences[1]
    b[0::2], b[1::2] = sequences[2], sequences[3]
    return np.stack((a, alt(a), b, alt(b)))


def deinterleave(sequences: Int8Array, source: FoldSource = "paley") -> Int8Array:
    """Fold a source-specific (4, 2m) binary state into a new (4, m) array.

    Paley mode splits the first two rows, which must form a negaperiodic Golay
    pair for the result to be a solution. Turyn mode recovers even columns;
    alt mode splits rows 0 and 2. These projections invert their respective
    lifts, not arbitrary GS4 states. Validate claimed solutions independently.
    """
    if (
        sequences.ndim != 2
        or sequences.shape[0] != 4
        or sequences.shape[1] == 0
        or sequences.shape[1] % 2
    ):
        raise ValueError("deinterleave expects shape (4, 2m), m > 0")
    if source == "turyn":
        return sequences[:, ::2].copy()
    if source not in ("paley", "alt"):
        raise ValueError(f"unknown source: {source}")
    second = 1 if source == "paley" else 2
    return np.stack(
        (sequences[0, ::2], sequences[0, 1::2], sequences[second, ::2], sequences[second, 1::2])
    )


def construct(order: int) -> Int8Array | None:
    """Return sequences from a Paley base and alt lifts, or None if unsupported.

    The argument is matrix order; the returned sequence length is order / 4.
    Full residual and matrix verification runs before returning a construction.
    """
    if order <= 0 or order % 8:
        return None
    base = order // 4
    doublings = 0
    while base >= 2 and base % 2 == 0:
        if supports_paley_ng(base):
            result = paley_ng_sequences(base)
            for _ in range(doublings):
                result = alt_double(result)
            verify_candidate(result)
            return result
        base //= 2
        doublings += 1
    return None


def verify_theorem(n: int, *, samples: int = 10) -> bool:
    """Sample NAF(alt(a), t) = (-1)^t NAF(a, t) at positive even length."""
    if n <= 0 or n % 2 or samples <= 0:
        raise ValueError("the alternation identity requires positive even n and samples > 0")
    rng = np.random.default_rng(42)
    signs = np.where(np.arange(n) % 2 == 0, 1, -1)
    for _ in range(samples):
        a = rng.choice((-1, 1), size=n).astype(np.int8)
        residual = Tracker._compute_residual(np.stack((a, a, a, a)))
        b = alt(a)
        alternating = Tracker._compute_residual(np.stack((b, b, b, b)))
        if not np.array_equal(alternating, signs * residual):
            return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Run construction, fold, round-trip or alternation controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("construct", help="Paley base plus alt doubling")
    build.add_argument("orders", nargs="+", type=int)
    paley = commands.add_parser("paley", help="Paley(2m) -> GS4(m), fully verified")
    paley.add_argument("lengths", nargs="+", type=int, metavar="M")
    trip = commands.add_parser("roundtrip", help="Bitwise inversion of both lifts")
    trip.add_argument("lengths", nargs="*", type=int, default=[3, 4, 13, 30])
    check = commands.add_parser("check", help="Sample the even-length alternation identity")
    check.add_argument("lengths", nargs="*", type=int, default=list(range(4, 33, 2)))
    sweep = commands.add_parser("sweep", help="Verify supported Paley folds up to max m")
    sweep.add_argument("--max-length", type=int, default=50)
    args = parser.parse_args(argv)
    try:
        if args.command == "construct":
            for order in args.orders:
                if construct(order) is None:
                    raise ValueError(f"order {order}: no supported Paley base")
                print(f"order={order}: verified")
        elif args.command in ("paley", "sweep"):
            if args.command == "sweep" and args.max_length <= 0:
                raise ValueError("max length must be positive")
            lengths = (
                args.lengths
                if args.command == "paley"
                else [m for m in range(1, args.max_length + 1) if supports_paley_ng(2 * m)]
            )
            for m in lengths:
                folded = deinterleave(paley_ng_sequences(2 * m))
                verify_candidate(folded)
                print(f"Paley({2 * m}) -> GS4({m}) -> H{4 * m}: verified")
        elif args.command == "roundtrip":
            rng = np.random.default_rng(42)
            for n in args.lengths:
                if n <= 0:
                    raise ValueError("length must be positive")
                for _ in range(20):
                    state = rng.choice((-1, 1), size=(4, n)).astype(np.int8)
                    lifts: tuple[tuple[FoldSource, Int8Array], ...] = (
                        ("turyn", double_gs4(state)),
                        ("alt", alt_double(state)),
                    )
                    for source, lifted in lifts:
                        if not np.array_equal(deinterleave(lifted, source), state):
                            raise ValueError(f"{source} round trip failed at n={n}")
                print(f"n={n}: both lifts invert bitwise (20 samples)")
        else:
            for n in args.lengths:
                if not verify_theorem(n):
                    raise ValueError(f"alternation identity failed at n={n}")
                print(f"n={n}: alternation identity passed (10 samples)")
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
