"""Advanced recovery: sieve distance, energy landscape, DR from perturbed."""
from __future__ import annotations

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from correlations import (nonperiodic_autocorrelation_state, nonperiodic_correlation_energy,
                           apply_nonperiodic_flip, nonperiodic_batch_energy)
from fixtures import tt36_sequences, turyn_lengths
from sieve import seed_turyn_batch
from fourier import project_weighted_nonperiodic_power

WEIGHTS_I = np.array((1, 1, 2, 2), dtype=np.int64)

SOL = tt36_sequences()
LENGTHS = turyn_lengths(36)


def _hamming(a, b):
    mask = np.zeros((4, 36), dtype=bool)
    for i, L in enumerate(LENGTHS):
        mask[i, :L] = True
    return int(np.sum(a[mask] != b[mask]))


def _energy(seq):
    return nonperiodic_correlation_energy(
        nonperiodic_autocorrelation_state(seq, lengths=LENGTHS, weights=WEIGHTS_I))


rng = np.random.default_rng(2026)
TOTAL = 143

# 1. Sieve vs Random distance
print("1) PSD-SIEVE vs RANDOM: DISTANCE TO TT(36) SOLUTION")
for count in [100, 1000, 10000]:
    sieved = seed_turyn_batch(36, count, rng)
    s_dist = [_hamming(seed, SOL) for seed in sieved]
    s_energy = [_energy(seed) for seed in sieved]
    random_batch = np.zeros((count, 4, 36), dtype=np.int8)
    for i in range(count):
        for ri, L in enumerate(LENGTHS):
            random_batch[i, ri, :L] = rng.choice([-1, 1], size=int(L))
        random_batch[i, 3, 35] = 0
    r_dist = [_hamming(seed, SOL) for seed in random_batch]
    r_energy = [_energy(seed) for seed in random_batch]
    print(f"\n{count} seeds:")
    print(f"  Sieved:  dist median={np.median(s_dist):.0f} min={min(s_dist)}  "
          f"energy median={np.median(s_energy):.0f} min={min(s_energy)}")
    print(f"  Random:  dist median={np.median(r_dist):.0f} min={min(r_dist)}  "
          f"energy median={np.median(r_energy):.0f} min={min(r_energy)}")

# 2. Energy vs noise
print("\n\n2) ENERGY vs NOISE LEVEL")
print(f"{'noise':>5} {'flips':>5} | {'start energy':>13} | {'recovery (greedy)':>16}")
print("-" * 50)
for pct in [0.5, 1, 2, 5, 10, 20, 50]:
    n_flips = max(1, int(TOTAL * pct / 100))
    energies_before, recovered = [], 0
    for trial in range(20):
        p = SOL.copy()
        for _ in range(n_flips):
            ri = int(rng.integers(0, 4)); ci = int(rng.integers(0, LENGTHS[ri]))
            p[ri, ci] *= -1
        e_before = _energy(p)
        energies_before.append(e_before)
        corr = nonperiodic_autocorrelation_state(p, lengths=LENGTHS, weights=WEIGHTS_I)
        best_e = e_before
        for _ in range(2000):
            improved = False
            for row in range(4):
                for col in range(int(LENGTHS[row])):
                    ne = apply_nonperiodic_flip(p, corr, row, col, lengths=LENGTHS, weight=int(WEIGHTS_I[row]))
                    if ne < best_e: best_e = ne; improved = True; break
                    apply_nonperiodic_flip(p, corr, row, col, lengths=LENGTHS, weight=int(WEIGHTS_I[row]))
                if improved: break
            if best_e == 0: recovered += 1; break
            if not improved: break
    print(f"{pct:>4}% {n_flips:>5} | e={np.median(energies_before):>10.0f} | {recovered:>4}/20")

# 3. Spectral DR from perturbed
print("\n\n3) SPECTRAL DR: START FROM PERTURBED SOLUTION")


def _dr_step(state):
    proj = project_weighted_nonperiodic_power(
        state, lengths=LENGTHS, weights=WEIGHTS_I.astype(np.float64), module=np)
    refl = 2 * proj - state
    signs = np.where(refl >= 0, 1.0, -1.0).astype(np.float32)
    signs[3, -1] = 0.0
    return (0.5 * (state + 2 * signs - refl)).astype(np.float32)


def _dr_energy(sf):
    signs = np.where(sf >= 0, 1.0, -1.0).astype(np.float32)
    signs[3, -1] = 0.0
    return float(nonperiodic_batch_energy(
        signs[None, ...], lengths=LENGTHS, weights=WEIGHTS_I, module=np)[0])


for perturb_pct in [1, 5, 10, 20]:
    n_flips = max(1, int(TOTAL * perturb_pct / 100))
    recovered = 0
    for trial in range(10):
        p = SOL.copy()
        for _ in range(n_flips):
            ri = int(rng.integers(0, 4)); ci = int(rng.integers(0, LENGTHS[ri]))
            p[ri, ci] *= -1
        state = p.astype(np.float32)
        best_e = _dr_energy(state)
        for inner_step in range(200):
            state = _dr_step(state)
            if inner_step % 10 == 9:
                e = _dr_energy(state)
                if e < best_e: best_e = e
                if e == 0: break
        if best_e == 0:
            recovered += 1
    print(f"  {perturb_pct:>3}% noise ({n_flips:>2} flips): {recovered}/10 recovered")
