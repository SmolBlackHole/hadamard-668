"""Recovery benchmark: how far can greedy repair reach at each scale?"""
from __future__ import annotations

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from correlations import (
    apply_nonperiodic_flip,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
)
from fixtures import TT8_SEQUENCES, TT36_HEX, tt8_sequences, tt36_sequences, turyn_lengths

WEIGHTS = np.array((1, 1, 2, 2), dtype=np.int64)


def _make_solution(sequences):
    n = max(len(s) for s in sequences)
    sol = np.zeros((4, n), dtype=np.int8)
    for i, s in enumerate(sequences):
        sol[i, :len(s)] = s
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    return sol, lengths


_SOLUTIONS: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _load_solution(n: int):
    if n in _SOLUTIONS:
        return _SOLUTIONS[n]
    if n == 8:
        sol, L = tt8_sequences(), turyn_lengths(8)
    elif n == 36:
        sol, L = tt36_sequences(), turyn_lengths(36)
    elif n <= 6:
        total = int(n * 3 + (n - 1))
        lengths = turyn_lengths(n)
        for bits in range(1 << total):
            seqs = [[], [], [], []]
            bit_pos = 0
            for si, length in enumerate(lengths):
                for _ in range(int(length)):
                    seqs[si].append(1 if (bits >> bit_pos) & 1 else -1)
                    bit_pos += 1
            sol, L = _make_solution(seqs)
            corr = nonperiodic_autocorrelation_state(sol, lengths=L, weights=WEIGHTS)
            if nonperiodic_correlation_energy(corr) == 0:
                _SOLUTIONS[n] = (sol, L)
                return sol, L
        return None
    else:
        return None
    _SOLUTIONS[n] = (sol, L)
    return sol, L


def _recover_greedy(perturbed, lengths, max_steps=3000):
    corr = nonperiodic_autocorrelation_state(perturbed, lengths=lengths, weights=WEIGHTS)
    current = perturbed.copy()
    best_e = nonperiodic_correlation_energy(corr)
    for _ in range(max_steps):
        improved = False
        for row in range(4):
            for col in range(int(lengths[row])):
                ne = apply_nonperiodic_flip(
                    current, corr, row, col, lengths=lengths, weight=int(WEIGHTS[row]))
                if ne < best_e:
                    best_e = ne
                    improved = True
                    break
                apply_nonperiodic_flip(
                    current, corr, row, col, lengths=lengths, weight=int(WEIGHTS[row]))
            if improved:
                break
        if best_e == 0:
            return True
        if not improved:
            return False
    return best_e == 0


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    print(f"{'n':>3} {'bits':>5} {'order':>6} | 0.5%  1%   2%   5%  10%  20%")
    print("-" * 50)

    for n in [2, 3, 4, 5, 6, 8, 36]:
        t0 = time.perf_counter()
        result = _load_solution(n)
        if result is None:
            print(f"{n:>3}   --   no TT({n}) solution exists")
            continue
        sol, lengths = result
        total_bits = int(lengths.sum())
        order = 4 * (3 * n - 1)
        row = f"{n:>3} {total_bits:>5} {order:>6} |"
        for pct in [0.5, 1, 2, 5, 10, 20]:
            n_flips = max(1, int(total_bits * pct / 100))
            recovered = 0
            for trial in range(20):
                p = sol.copy()
                for _ in range(n_flips):
                    ri = int(rng.integers(0, 4))
                    ci = int(rng.integers(0, lengths[ri]))
                    p[ri, ci] *= -1
                if _recover_greedy(p, lengths):
                    recovered += 1
            row += f"  {recovered:>2} "
        elapsed = time.perf_counter() - t0
        row += f" ({elapsed:.1f}s)"
        print(row)
