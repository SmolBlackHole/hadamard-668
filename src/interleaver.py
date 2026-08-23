"""Turyn-x2 doubling in transparent interleave/alternation normal form.

Self-contained Hadamard construction: Paley base + repeated doubling.
The doubling step is the Turyn-x2 lift expressed in coordinates that make
the embedded landscape visible: a = interleave(a0,a1), then (a, alt(a)).

This is equivalent to double_gs4 (Turyn) up to reversal and sign flips
on two sequences.  The value is the transparent landscape structure:
NAF(a, 2s) = NAF(a0, s) + NAF(a1, s), Q_2m = 4*Q_m, isomorphic move graphs.

Usage::

    python alt-double.py 16          # order 64
    python alt-double.py 32 64 128   # multiple orders
    python alt-double.py --check     # verify NAF(alt(a))(t) = (-1)^t * NAF(a)(t)
    python alt-double.py --sweep     # Paley -> doubling chain
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Int8 = npt.NDArray[np.int8]
Int32 = npt.NDArray[np.int32]

# ============================================================
# 1. NAF computation (negaperiodic autocorrelation)
# ============================================================


def naf_of(seqs: Int8) -> Int32:
    """NAF of k sequences of length n: shape (k, n) -> (n,)."""
    n = seqs.shape[1]
    j = np.arange(n, dtype=np.intp)[:, None]
    t = np.arange(n, dtype=np.intp)[None, :]
    shifted = (j + t) % n
    sign = np.where(j + t < n, 1, -1).astype(np.int32)
    values = seqs.astype(np.int32)
    return np.sum(values[:, :, None] * values[:, shifted] * sign, axis=(0, 1))


def gs4_energy(seqs: Int8) -> int:
    """GS4 residual energy: 0 means Hadamard solution."""
    n = seqs.shape[1]
    r = naf_of(seqs)
    m = (n - 1) // 2  # independent lags
    u = r[1 : m + 1] // 4
    return int(64 * n * np.dot(u, u))


# ============================================================
# 2. Alt-Operator and doubling lift
# ============================================================


def alt(a: Int8) -> Int8:
    """Alt-Operator: alt(a)[j] = a[j] * (-1)^j."""
    factor = np.where(np.arange(len(a)) % 2 == 0, 1, -1).astype(np.int8)
    return (a * factor).astype(np.int8)


def alt_double(sequences: Int8) -> Int8:
    """Turyn x2 doubling in interleave/alternation normal form.

    Input:  4 x m array of +-1, with gs4_energy() == 0.
    Output: 4 x 2m array of +-1, with gs4_energy() == 0.

    Construction:
        a = interleave(seqs[0], seqs[1])
        b = interleave(seqs[2], seqs[3])
        result = (a, alt(a), b, alt(b))

    Equivalent to double_gs4 up to reversal/sign on two sequences.
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


# ============================================================
# 3. Ito-Paley construction (quadratic character over GF(p^2))
# ============================================================


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
    """True when the Paley/Ito construction works (2n-1 prime, n even)."""
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
    """Construct GS4(n) via Ito-Paley.  Requires supports_paley_ng(n).

    Returns 4 x n array of +/-1.
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


# ============================================================
# 4. Hadamard matrix construction (Goethals-Seidel array)
# ============================================================


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
    """Build the full 4n x 4n Hadamard matrix via Goethals-Seidel.

    Input:  four sequences of length n (GS4 condition satisfied).
    Output: 4n x 4n Hadamard matrix (entries +/-1, HH^T = 4nI).
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


# ============================================================
# 5. Verification
# ============================================================


def verify_hadamard(H: Int8) -> bool:
    """Check H * H^T == N * I."""
    N = H.shape[0]
    HHt = H.astype(np.int64) @ H.astype(np.int64).T
    expected = np.eye(N, dtype=np.int64) * N
    return bool(np.array_equal(HHt, expected))


# ============================================================
# 6. Theorem verification (for correctness checks)
# ============================================================


def verify_theorem(n: int) -> bool:
    """Check: NAF(alt(a))(t) == (-1)^t * NAF(a)(t) for all sequences."""
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


# ============================================================
# 7. Main
# ============================================================


def construct(order: int) -> Int8 | None:
    """Build a Hadamard matrix of given order via Paley + Alt-chain.

    Returns None if the order can't be constructed this way.
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
    import argparse

    p = argparse.ArgumentParser(description="Alt-doubling Hadamard construction (self-contained)")
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
