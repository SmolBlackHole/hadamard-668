"""Deinterleave: negacyclic Golay pair at 2m -> GS4(m).

For any NG pair (a,b) at even length 2m, the even-lag NAF decomposition
NAF(2s) = NAF_even(s) + NAF_odd(s) implies that (a_even, a_odd, b_even, b_odd)
is GS4(m).  This includes Paley as a special case.

Self-contained. No repo imports.

Usage::

    python deinterleave.py --paley 83      # Paley(166) -> GS4(83) -> H332
    python deinterleave.py --turyn 4       # Turyn round-trip test
    python deinterleave.py --sweep         # all constructable n
    python deinterleave.py --verify        # round-trip: deint(lift(x)) == x
"""

from __future__ import annotations

import numpy as np

# ============================================================
# Paley/Ito (same as alt-double.py, src/constructions.py)
# ============================================================


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _prime_factors(n: int) -> list[int]:
    factors: list[int] = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return factors


def supports_paley(n: int) -> bool:
    return n > 0 and n % 2 == 0 and _is_prime(2 * n - 1)


GF2 = tuple[int, int]


def _gf_mul(a: int, b: int, d: int, e: int, c: int, prime: int) -> GF2:
    return ((a * d - c * b * e) % prime, (a * e + b * d - b * e) % prime)


def _gf_pow(base: GF2, exp: int, c: int, prime: int) -> GF2:
    a, b = base
    r0: int = 1
    r1: int = 0
    while exp:
        if exp & 1:
            r0, r1 = _gf_mul(r0, r1, a, b, c, prime)
        a, b = _gf_mul(a, b, a, b, c, prime)
        exp >>= 1
    return (r0, r1)


def _gf_primitive(prime: int, factors: list[int]) -> tuple[GF2, int]:
    """Find primitive element of GF(prime^2). Returns (generator, coefficient)."""
    field_order = prime * prime
    for c in range(prime):
        disc = (1 - 4 * c) % prime
        if pow(disc, (prime - 1) // 2, prime) != prime - 1:
            continue
        cand: GF2 = (0, 1)
        ok = True
        for f in factors:
            if _gf_pow(cand, (field_order - 1) // f, c, prime) == (1, 0):
                ok = False
                break
        if ok:
            return cand, c
    raise RuntimeError(f"no primitive element for GF({prime}^2)")


def paley_gs4(n: int) -> np.ndarray:
    """Build GS4(n) via Ito-Paley."""
    prime = 2 * n - 1
    factors = _prime_factors(prime * prime - 1)

    gen, coeff = _gf_primitive(prime, factors)
    residues: set[int] = {v * v % prime for v in range(1, prime)}
    alpha: GF2 = _gf_pow(gen, n, coeff, prime)
    w: GF2 = _gf_pow(gen, n - 1, coeff, prime)

    def _sign(exp: int) -> np.int8:
        z = _gf_pow(w, exp, coeff, prime)
        tv, tx = _gf_mul(alpha[0], alpha[1], z[0], z[1], coeff, prime)
        return np.int8(1 if (2 * tv - tx) % prime in residues else -1)

    a = np.fromiter((_sign(2 * i) for i in range(n)), dtype=np.int8, count=n)
    b = np.fromiter((_sign(2 * i + 1) for i in range(n)), dtype=np.int8, count=n)
    return np.stack((a, b, a, b))


# ============================================================
# Turyn doubling
# ============================================================


def turyn_double(sequences: np.ndarray) -> np.ndarray:
    """Turyn x2: GS4(m) -> GS4(2m) via Golay-2 factor."""
    p = np.asarray((1, -1), dtype=np.int8)  # Golay pair
    q = np.asarray((1, 1), dtype=np.int8)
    plus = (p + q) // 2  # [1, 0]
    minus = (p - q) // 2  # [0, -1]
    plus16 = plus.astype(np.int16)
    minus16 = minus.astype(np.int16)
    result_parts: list[np.ndarray] = []
    for pair_idx in range(2):
        c = sequences[2 * pair_idx].astype(np.int16)
        d = sequences[2 * pair_idx + 1].astype(np.int16)
        e = plus16[:, None] * c + minus16[:, None] * d[::-1]
        f = plus16[:, None] * d - minus16[:, None] * c[::-1]
        result_parts.append(e.reshape(-1, order="F").astype(np.int8))
        result_parts.append(f.reshape(-1, order="F").astype(np.int8))
    return np.stack(result_parts)


def _alt_seq(seq: np.ndarray) -> np.ndarray:
    """Alt-Operator: seq[j] * (-1)^j."""
    n = len(seq)
    factor = np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8)
    return (seq * factor).astype(np.int8)


def alt_double(sequences: np.ndarray) -> np.ndarray:
    """Alt doubling: GS4(m) -> GS4(2m)."""
    m = sequences.shape[1]
    a_out = np.zeros(2 * m, dtype=np.int8)
    b_out = np.zeros(2 * m, dtype=np.int8)
    a_out[0::2] = sequences[0]
    a_out[1::2] = sequences[1]
    b_out[0::2] = sequences[2]
    b_out[1::2] = sequences[3]
    return np.stack((a_out, _alt_seq(a_out), b_out, _alt_seq(b_out)))


# ============================================================
# Deinterleaving
# ============================================================


def deinterleave_paley(gs4_2m: np.ndarray) -> np.ndarray:
    """NG(2m) -> GS4(m): (a_even, a_odd, b_even, b_odd).

    For any negaperiodic Golay pair (a,b) at length 2m, the even-lag
    NAF decomposition NAF(2s) = NAF_even(s) + NAF_odd(s) implies that
    the subsampled quadruple is GS4(m).

    In particular, when the input is Paley(2m) the output is GS4(m).
    """
    return np.stack(
        [
            gs4_2m[0, 0::2],
            gs4_2m[0, 1::2],
            gs4_2m[1, 0::2],
            gs4_2m[1, 1::2],
        ]
    )


def deinterleave_turyn(gs4_2m: np.ndarray) -> np.ndarray:
    """Turyn(GS4(m)) -> GS4(m): even columns of all 4 sequences.

    Turyn formula: e[2j]=a[j], f[2j]=b[j]. Even positions = original.
    """
    return np.stack([gs4_2m[s, 0::2] for s in range(4)])


def deinterleave_alt(gs4_2m: np.ndarray) -> np.ndarray:
    """Alt(GS4(m)) -> GS4(m): subsample even positions.

    Alt formula: a=c0|c1, alt(a)=c0|(-c1). Both have c0 at even j.
    So even positions of ALL 4 sequences recover (c0, c0, d0, d0).
    Actually: even seqs[0][::2] = a0, seqs[1][::2] = a0, etc.
    The correct recovery is (seqs[0][::2], seqs[0][1::2], seqs[2][::2], seqs[2][1::2]).
    """
    return np.stack(
        [
            gs4_2m[0, 0::2],
            gs4_2m[0, 1::2],
            gs4_2m[2, 0::2],
            gs4_2m[2, 1::2],
        ]
    )


def deinterleave(gs4_2m: np.ndarray, source: str = "paley") -> np.ndarray:
    """Deinterleave GS4(2m) -> GS4(m) based on the source construction."""
    if source == "paley":
        return deinterleave_paley(gs4_2m)
    if source == "turyn":
        return deinterleave_turyn(gs4_2m)
    if source == "alt":
        return deinterleave_alt(gs4_2m)
    raise ValueError(f"unknown source: {source}")


# ============================================================
# NAF energy
# ============================================================


def gs4_energy(seqs: np.ndarray) -> int:
    n = seqs.shape[1]
    j = np.arange(n, dtype=np.intp)[:, None]
    t = np.arange(n, dtype=np.intp)[None, :]
    shifted = (j + t) % n
    sign = np.where(j + t < n, 1, -1).astype(np.int32)
    values = seqs.astype(np.int32)
    r = np.sum(values[:, :, None] * values[:, shifted] * sign, axis=(0, 1))
    m_eff = (n - 1) // 2
    u = r[1 : m_eff + 1] // 4
    return int(64 * n * np.dot(u, u))


# ============================================================
# Hadamard builder
# ============================================================


def _negacirc(values: np.ndarray) -> np.ndarray:
    n = len(values)
    idx = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
    sgn = np.where(np.arange(n)[None, :] >= np.arange(n)[:, None], np.int8(1), np.int8(-1))
    return sgn * values[idx]


def build_hadamard(seqs: np.ndarray) -> np.ndarray:
    A = _negacirc(seqs[0])
    B = _negacirc(seqs[1])
    C = _negacirc(seqs[2])
    D = _negacirc(seqs[3])
    return np.block(
        [
            [A, B[:, ::-1], C[:, ::-1], D[:, ::-1]],
            [-B[:, ::-1], A, -D.T[:, ::-1], C.T[:, ::-1]],
            [-C[:, ::-1], D.T[:, ::-1], A, -B.T[:, ::-1]],
            [-D[:, ::-1], -C.T[:, ::-1], B.T[:, ::-1], A],
        ]
    )


def verify(mat: np.ndarray) -> bool:
    N = mat.shape[0]
    HHt = mat.astype(np.int64) @ mat.astype(np.int64).T
    return bool(np.array_equal(HHt, np.eye(N, dtype=np.int64) * N))


# ============================================================
# Main
# ============================================================


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Deinterleave: Paley(2m) -> GS4(m), Turyn/Alt round-trips"
    )
    p.add_argument(
        "--paley", type=int, metavar="M", help="Construct GS4(m) via Paley(2m) + deinterleave"
    )
    p.add_argument(
        "--turyn",
        type=int,
        metavar="M",
        help="Test Turyn round-trip: GS4(m) -> Turyn -> deint -> GS4(m)",
    )
    p.add_argument(
        "--alt", type=int, metavar="M", help="Test Alt round-trip: GS4(m) -> Alt -> deint -> GS4(m)"
    )
    p.add_argument(
        "--sweep", action="store_true", help="Sweep all deinterleavable Paley(2m) -> GS4(m)"
    )
    p.add_argument(
        "--verify", action="store_true", help="Verify round-trip: deint(lift(x)) == x"
    )
    args = p.parse_args()

    if args.verify:
        print("=== Round-trip verification: deint(lift(x)) == x ===\n")
        rng = np.random.default_rng(42)
        for source in ["turyn", "alt"]:
            ok = 0
            total = 0
            for _ in range(20):
                a = rng.choice([-1, 1], size=6).astype(np.int8)
                b = rng.choice([-1, 1], size=6).astype(np.int8)
                x = np.stack([a, b, a, b])
                lift_fn = turyn_double if source == "turyn" else alt_double
                lifted = lift_fn(x)
                back = deinterleave(lifted, source)
                total += 1
                if source == "turyn":
                    ok += int(np.array_equal(back, x))
                else:
                    ok += int(gs4_energy(back) == gs4_energy(x))
            kind = "bitwise" if source == "turyn" else "energy"
            print(f"  {source}: {ok}/{total} ({kind})")
        return

    if args.sweep:
        print("=== Paley(2m) -> deinterleave -> GS4(m) ===\n")
        print(f"{'2m':>4} {'m':>4} {'odd?':>5} {'Q':>6} {'order':>6}")
        print("-" * 32)
        count = 0
        for m in range(1, 200):
            n_even = 2 * m
            if not supports_paley(n_even):
                continue
            gs4 = paley_gs4(n_even)
            deint = deinterleave_paley(gs4)
            q = gs4_energy(deint) // (64 * m) if m > 0 else 0
            mat = build_hadamard(deint)
            ok = verify(mat)
            count += 1
            print(
                f"{n_even:>4} {m:>4} {'Ja' if m % 2 else 'Nein':>5} "
                f"{q:>6} {4 * m:>6} {'OK' if ok else 'FAIL'}"
            )
        print(f"\n{count} Hadamard orders constructed")
        return

    if args.paley is not None:
        m = args.paley
        n_even = 2 * m
        if not supports_paley(n_even):
            print(f"m={m}: 2m={n_even}, 2*2m-1={4 * m - 1} not prime")
            return
        gs4 = paley_gs4(n_even)
        deint = deinterleave_paley(gs4)
        q = gs4_energy(deint) // (64 * m) if m > 0 else 0
        mat = build_hadamard(deint)
        ok = verify(mat)
        print(f"Paley({n_even}) -> GS4({m}) -> H{4 * m}: Q={q}, {'OK' if ok else 'FAIL'}")
        return

    if args.turyn is not None:
        m = args.turyn
        rng = np.random.default_rng(42)
        a = rng.choice([-1, 1], size=m).astype(np.int8)
        b = rng.choice([-1, 1], size=m).astype(np.int8)
        x = np.stack([a, b, a, b])
        lifted = turyn_double(x)
        back = deinterleave_turyn(lifted)
        e_orig = gs4_energy(x)
        e_back = gs4_energy(back)
        print(
            f"Turyn round-trip m={m}: original E={e_orig}, "
            f"deint(turyn(x)) E={e_back} {'OK' if e_orig == e_back else 'FAIL'}"
        )
        return

    if args.alt is not None:
        m = args.alt
        rng = np.random.default_rng(42)
        a = rng.choice([-1, 1], size=m).astype(np.int8)
        b = rng.choice([-1, 1], size=m).astype(np.int8)
        x = np.stack([a, b, a, b])
        lifted = alt_double(x)
        back = deinterleave_alt(lifted)
        e_orig = gs4_energy(x)
        e_back = gs4_energy(back)
        print(
            f"Alt round-trip m={m}: original E={e_orig}, "
            f"deint(alt(x)) E={e_back} {'OK' if e_orig == e_back else 'FAIL'}"
        )
        return

    p.print_help()


if __name__ == "__main__":
    main()
