"""Explore Turyn-x2 doubling in interleave/alternation normal form.

This standalone research CLI is not part of the canonical construction or
search pipeline. It provides a self-contained Paley base plus repeated
doubling for inspecting the embedded landscape.

The doubling step is the Turyn-x2 lift expressed in coordinates that make
the embedded landscape visible: a = interleave(a0,a1), then (a, alt(a)).

This is equivalent to double_gs4 (Turyn) up to reversal and sign flips
on two sequences.  The value is the transparent landscape structure:
NAF(a, 2s) = NAF(a0, s) + NAF(a1, s), Q_2m = 4*Q_m, isomorphic move graphs.

Usage::

    python -m src.interleaver 16          # order 16
    python -m src.interleaver 32 64 128   # multiple orders
    python -m src.interleaver --check     # verify the alternation identity
    python -m src.interleaver --sweep     # Paley -> doubling chain
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Int8 = npt.NDArray[np.int8]
Int32 = npt.NDArray[np.int32]

# --- 1. NAF computation (negaperiodic autocorrelation) ------------------------


def naf_of(seqs: Int8) -> Int32:
    """Compute the summed negaperiodic autocorrelation vector.

    Args:
        seqs: Binary sequences with shape ``(k, n)``.

    Returns:
        Summed NAF values for lags ``0`` through ``n - 1``.
    """
    n = seqs.shape[1]
    j = np.arange(n, dtype=np.intp)[:, None]
    t = np.arange(n, dtype=np.intp)[None, :]
    shifted = (j + t) % n
    sign = np.where(j + t < n, 1, -1).astype(np.int32)
    values = seqs.astype(np.int32)
    return np.sum(values[:, :, None] * values[:, shifted] * sign, axis=(0, 1))


def gs4_energy(seqs: Int8) -> int:
    """Compute exact GS4 matrix energy from independent NAF residuals.

    Args:
        seqs: Four binary sequences with shape ``(4, n)``.

    Returns:
        Energy ``64 * n * Q``. Zero identifies a GS4 solution.
    """
    n = seqs.shape[1]
    r = naf_of(seqs)
    m = (n - 1) // 2  # independent lags
    u = r[1 : m + 1] // 4
    return int(64 * n * np.dot(u, u))


# --- 2. Alt-Operator and doubling lift ----------------------------------------


def alt(a: Int8) -> Int8:
    """Apply alternating signs to a sequence.

    Args:
        a: One sequence of length ``n``.

    Returns:
        A new sequence whose entry ``j`` is ``a[j] * (-1)**j``.
    """
    factor = np.where(np.arange(len(a)) % 2 == 0, 1, -1).astype(np.int8)
    return (a * factor).astype(np.int8)


def alt_double(sequences: Int8) -> Int8:
    """Turyn x2 doubling in interleave/alternation normal form.

    Args:
        sequences: Four length-``m`` binary sequences with zero GS4 energy.

    Returns:
        Four length-``2m`` binary sequences with zero GS4 energy.

    Raises:
        ValueError: If ``sequences`` does not have four rows.

    Note:
        The construction is equivalent to ``double_gs4`` up to reversal and
        sign changes on two sequences.
    """
    if sequences.ndim != 2 or sequences.shape[0] != 4:
        raise ValueError("alt_double expects shape (4, m)")
    m = sequences.shape[1]
    a_out = np.zeros(2 * m, dtype=np.int8)
    b_out = np.zeros(2 * m, dtype=np.int8)
    a_out[0::2] = sequences[0]
    a_out[1::2] = sequences[1]
    b_out[0::2] = sequences[2]
    b_out[1::2] = sequences[3]
    return np.stack((a_out, alt(a_out), b_out, alt(b_out)))


# --- 3. Ito-Paley construction (quadratic character over GF(p^2)) -------------


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    d = 3
    while d * d <= value:
        if value % d == 0:
            return False
        d += 2
    return True


def _prime_factors(value: int) -> list[int]:
    factors: list[int] = []
    d = 2
    while d * d <= value:
        if value % d == 0:
            factors.append(d)
            while value % d == 0:
                value //= d
        d += 1 if d == 2 else 2
    if value > 1:
        factors.append(value)
    return factors


def supports_paley_ng(n: int) -> bool:
    """Return whether ``n`` satisfies the Ito-Paley preconditions."""
    return n > 0 and n % 2 == 0 and _is_prime(2 * n - 1)


def _gf_mul(a: int, b: int, d: int, e: int, c: int, prime: int) -> tuple[int, int]:
    """Multiply (a+bi)*(d+ei) in GF(p^2) with i^2 = c mod p."""
    return ((a * d - c * b * e) % prime, (a * e + b * d - b * e) % prime)


def _gf_pow(base: tuple[int, int], exp: int, c: int, prime: int) -> tuple[int, int]:
    """Exponentiation in GF(p^2)*."""
    a, b = base
    result: tuple[int, int] = (1, 0)
    while exp:
        if exp & 1:
            ra, rb = result
            result = _gf_mul(ra, rb, a, b, c, prime)
        a, b = _gf_mul(a, b, a, b, c, prime)
        exp >>= 1
    return result


def paley_ng(n: int) -> Int8:
    """Construct a GS4 quadruple using the Ito-Paley construction.

    Args:
        n: Even sequence length such that ``2 * n - 1`` is prime.

    Returns:
        Four binary sequences with shape ``(4, n)``.

    Raises:
        ValueError: If ``n`` does not satisfy the construction conditions.
        RuntimeError: If no primitive field generator can be found.
    """
    if not supports_paley_ng(n):
        raise ValueError(f"paley_ng: n={n} needs 2n-1 prime and n even")

    prime = 2 * n - 1
    field_order = prime * prime
    factors = _prime_factors(field_order - 1)

    generator: tuple[int, int] | None = None
    coefficient = 0
    for c in range(prime):
        disc = (1 - 4 * c) % prime
        if pow(disc, (prime - 1) // 2, prime) != prime - 1:
            continue
        cand: tuple[int, int] = (0, 1)
        if all(_gf_pow(cand, (field_order - 1) // f, c, prime) != (1, 0) for f in factors):
            generator = cand
            coefficient = c
            break

    if generator is None:
        raise RuntimeError(f"no primitive GF({prime}^2) generator found")

    residues = {v * v % prime for v in range(1, prime)}
    alpha = _gf_pow(generator, n, coefficient, prime)
    w = _gf_pow(generator, n - 1, coefficient, prime)

    def _sign(exp: int) -> np.int8:
        z = _gf_pow(w, exp, coefficient, prime)
        tv, tx = _gf_mul(alpha[0], alpha[1], z[0], z[1], coefficient, prime)
        return np.int8(1 if (2 * tv - tx) % prime in residues else -1)

    a = np.fromiter((_sign(2 * i) for i in range(n)), dtype=np.int8, count=n)
    b = np.fromiter((_sign(2 * i + 1) for i in range(n)), dtype=np.int8, count=n)
    return np.stack((a, b, a, b))


# --- 4. Hadamard matrix construction (Goethals-Seidel array) ------------------


def _negacirculant(values: Int8) -> Int8:
    """Build n x n negacirculant matrix from a sequence of length n.

    Entry (i,j) = sign(i,j) * values[(j-i) mod n]
    where sign(i,j) = +1 if j >= i, -1 if j < i.
    """
    n = len(values)
    idx = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
    sign = np.where(np.arange(n)[None, :] >= np.arange(n)[:, None], np.int8(1), np.int8(-1))
    return sign * values[idx]


def build_hadamard(sequences: Int8) -> Int8:
    """Build a Goethals-Seidel matrix from four negacyclic sequences.

    Args:
        sequences: Four length-``n`` sequences satisfying the GS4 condition.

    Returns:
        Binary matrix with shape ``(4n, 4n)``.
    """
    A = _negacirculant(sequences[0])
    B = _negacirculant(sequences[1])
    C = _negacirculant(sequences[2])
    D = _negacirculant(sequences[3])

    BR = B[:, ::-1]
    CR = C[:, ::-1]
    DR = D[:, ::-1]
    BtR = B.T[:, ::-1]
    CtR = C.T[:, ::-1]
    DtR = D.T[:, ::-1]

    return np.block(
        [
            [A, BR, CR, DR],
            [-BR, A, -DtR, CtR],
            [-CR, DtR, A, -BtR],
            [-DR, -CtR, BtR, A],
        ]
    )


# --- 5. Verification ----------------------------------------------------------


def verify_hadamard(H: Int8) -> bool:
    """Check the exact Hadamard Gram identity.

    Args:
        H: Candidate square matrix.

    Returns:
        Whether ``H @ H.T`` equals its order times the identity matrix.
    """
    N = H.shape[0]
    HHt = H.astype(np.int64) @ H.astype(np.int64).T
    expected = np.eye(N, dtype=np.int64) * N
    return bool(np.array_equal(HHt, expected))


# --- 6. Theorem verification (for correctness checks) -------------------------


def verify_theorem(n: int) -> bool:
    """Numerically check the NAF alternation identity on random sequences.

    Args:
        n: Even sequence length to sample.

    Returns:
        Whether all sampled lags satisfy the identity. Odd lengths return
        ``False`` because the tested identity does not apply.
    """
    if n % 2:
        print(f"  n={n}: odd -> theorem does NOT hold (expected)")
        return False
    rng = np.random.default_rng(42)
    for _ in range(10):
        a = rng.choice([-1, 1], size=n).astype(np.int8)
        aa = alt(a)
        naf_a = naf_of(np.stack([a, a, a, a]))
        naf_aa = naf_of(np.stack([aa, aa, aa, aa]))
        for t in range(1, n):
            if naf_aa[t] != ((-1) ** t) * naf_a[t]:
                return False
    return True


# --- 7. Main ------------------------------------------------------------------


def construct(order: int) -> Int8 | None:
    """Construct a Hadamard matrix through a Paley base and doubling chain.

    Args:
        order: Requested matrix order, divisible by four.

    Returns:
        The constructed matrix, or ``None`` when this experimental chain does
        not support the order.
    """
    if order <= 0 or order % 4 != 0:
        return None

    n = order // 4
    if n % 2:
        return None  # alt-double needs even n

    # Find a Paley base by repeatedly halving n
    base_n = n
    doublings = 0
    while base_n % 2 == 0 and base_n >= 2:
        if supports_paley_ng(base_n):
            gs4 = paley_ng(base_n)
            for _ in range(doublings):
                gs4 = alt_double(gs4)
            # Check GS4 condition
            if gs4_energy(gs4) != 0:
                return None
            return build_hadamard(gs4)
        base_n //= 2
        doublings += 1

    return None  # no Paley base found


def main() -> None:
    """Run the standalone construction and identity-check experiments."""
    import argparse

    p = argparse.ArgumentParser(
        description="Standalone experimental alt-doubling Hadamard construction"
    )
    p.add_argument("orders", nargs="*", type=int, help="Hadamard orders to construct")
    p.add_argument("--check", action="store_true", help="Verify alt-theorem numerically")
    p.add_argument("--sweep", action="store_true", help="Sweep Paley -> alt-chain")
    args = p.parse_args()

    if args.check:
        print("=== Alt-theorem verification ===\n")
        for n in range(4, 33, 2):
            ok = verify_theorem(n)
            print(f"  n={n:>3}: {'OK' if ok else 'FAIL'}")
        return

    if args.sweep:
        print("=== Paley -> Alt-chain sweep (pure construction) ===\n")
        print(f"{'base':>5}  {'doublings':>9}  {'n_final':>7}  {'order':>6}  {'OK?':>5}")
        print("-" * 42)
        for base_n in [2, 4, 6, 12, 18, 30, 52]:
            if not supports_paley_ng(base_n):
                continue
            for dbl in range(6):
                fn = base_n * (2**dbl)
                order = 4 * fn
                gs4 = paley_ng(base_n)
                for _ in range(dbl):
                    gs4 = alt_double(gs4)
                ok = gs4_energy(gs4) == 0
                if not ok:
                    break
                hmat = build_hadamard(gs4)
                vfy = verify_hadamard(hmat)
                print(f"{base_n:>5}  {dbl:>9}  {fn:>7}  {order:>6}  {'OK' if vfy else 'FAIL':>5}")
        return

    if not args.orders:
        p.print_help()
        return

    for order in args.orders:
        mat = construct(order)
        if mat is None:
            print(f"order {order}: cannot construct (no Paley base or n=order/4 is odd)")
            continue
        vfy = verify_hadamard(mat)
        nnz = np.count_nonzero(mat)
        print(
            f"order {order}: {'OK' if vfy else 'FAIL'}, "
            f"shape={mat.shape}, entries={nnz} "
            f"({100 * nnz / mat.size:.0f}% nonzero)"
        )


if __name__ == "__main__":
    main()
