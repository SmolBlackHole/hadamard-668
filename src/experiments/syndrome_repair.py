"""GPU-parallel repair: single, pair, triple flips.

Evaluates all single-flip candidates via batched GPU FFT, periodically
scans all pairs for non-greedy jumps, and samples triple flips from
gradient-ranked positions.
"""
from __future__ import annotations
from correlations import (nonperiodic_autocorrelation_state,
                          nonperiodic_correlation_energy,
                          apply_nonperiodic_flip)
from gpu import xp
import numpy as np

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


WEIGHTS_I = np.array((1, 1, 2, 2), dtype=np.int64)
WEIGHTS_F = np.array((1, 1, 2, 2), dtype=np.float64)


def _residual_gpu(seq_batch, lengths, weights_f):
    x_seq = xp.asarray(seq_batch, dtype=xp.float32)
    n = int(lengths[0])
    fft_size = 2 * n - 1
    spectrum = xp.fft.fft(x_seq, n=fft_size, axis=2)
    corr = xp.fft.ifft(xp.abs(spectrum) ** 2, axis=2).real
    w = xp.asarray(weights_f, dtype=xp.float32)
    total = xp.sum(w[None, :, None] * corr, axis=1)
    return total[:, 1:n]


def _batch_flip_energy_gpu(seq, lengths, flip_sets):
    B = len(flip_sets)
    if B == 0:
        return xp.array([], dtype=xp.float64)
    batch = xp.repeat(seq[None, ...], B, axis=0)
    for idx, flips in enumerate(flip_sets):
        for r, c in flips:
            batch[idx, r, c] *= -1
    residuals = _residual_gpu(batch, lengths, WEIGHTS_F)
    return xp.sum(residuals ** 2, axis=1)


def _fft_gradient(seq_float, lengths):
    n = int(lengths[0])
    fft_size = 2 * n - 1
    spectrum = xp.fft.fft(seq_float, n=fft_size, axis=1)
    corr_g = xp.fft.ifft(xp.abs(spectrum) ** 2, axis=1).real
    w = xp.asarray(WEIGHTS_F, dtype=corr_g.dtype)
    total_g = xp.sum(w[:, None] * corr_g, axis=0)
    coeff = xp.zeros(fft_size, dtype=corr_g.dtype)
    coeff[1:n] = total_g[1:n]
    coeff[-(n - 1):] = total_g[1:n][::-1]
    grad = (2.0 * w[:, None] *
            xp.fft.ifft(xp.fft.fft(coeff)[None, :] * spectrum, axis=1).real[:, :n])
    return grad


class GpuRepair:
    def __init__(self, sequences, lengths):
        self.seq = sequences.copy()
        self.lengths = lengths
        self.n = int(lengths[0])
        self.all_positions = [(r, c) for r in range(4)
                              for c in range(int(lengths[r]))]
        self.n_pos = len(self.all_positions)
        self.pairs = [(self.all_positions[i], self.all_positions[j])
                      for i in range(self.n_pos) for j in range(i + 1, self.n_pos)]

    def _exact_energy(self):
        corr = nonperiodic_autocorrelation_state(
            self.seq, lengths=self.lengths, weights=WEIGHTS_I)
        return float(nonperiodic_correlation_energy(corr))

    def _gradient_top_positions(self, n_top=60):
        seq_f = xp.asarray(self.seq.astype(np.float64), dtype=xp.float64)
        grad = _fft_gradient(seq_f, self.lengths)
        score = xp.float64(-2.0) * xp.asarray(self.seq,
                                              dtype=xp.float64) * grad
        score[3, int(self.lengths[3]):] = xp.inf
        flat = score.ravel()
        indices = xp.argsort(flat)[:n_top]
        return [self.all_positions[int(idx)] for idx in xp.asnumpy(indices)
                if int(idx) < self.n_pos]

    def _sample_triples(self, positions, n_triples=4096):
        n = len(positions)
        if n < 3:
            return []
        n_pick = min(n_triples, n * (n - 1) * (n - 2) // 6)
        triples = []
        rng = np.random.default_rng(int(time.perf_counter_ns() % (2**31)))
        seen = set()
        for _ in range(n_pick * 3):
            i, j, k = (int(v) for v in rng.integers(0, n, 3))
            if i == j or j == k or i == k:
                continue
            key = tuple(sorted([i, j, k]))
            if key in seen:
                continue
            seen.add(key)
            triples.append([positions[i], positions[j], positions[k]])
            if len(triples) >= n_pick:
                break
        return triples

    def repair(self, max_steps=200, pair_interval=3, triple_interval=8,
               max_batch=16384):
        all_flips = []
        best_e = self._exact_energy()
        if best_e == 0:
            return [], 0.0

        seq_f = xp.asarray(self.seq.astype(np.float32), dtype=xp.float32)
        current_e = best_e

        for step in range(max_steps):
            improved = False

            # Phase 1: ALL single flips
            n_chunks = (self.n_pos + max_batch - 1) // max_batch
            best_single_e = float("inf")
            best_single_pos = None
            for ci in range(n_chunks):
                chunk = self.all_positions[ci * max_batch:(ci + 1) * max_batch]
                energies = _batch_flip_energy_gpu(
                    seq_f, self.lengths, [[p] for p in chunk])
                local_best = int(xp.argmin(energies))
                if float(energies[local_best]) < best_single_e:
                    best_single_e = float(energies[local_best])
                    best_single_pos = chunk[local_best]

            if best_single_e < current_e:
                r, c = best_single_pos
                self.seq[r, c] *= -1
                seq_f[r, c] *= -1
                current_e = best_single_e
                all_flips.append((r, c))
                improved = True
                if current_e < 1e-6 or self._exact_energy() == 0:
                    return all_flips, 0.0

            # Phase 2: Pair evaluation
            if step % pair_interval == pair_interval - 1 and current_e > 1e-6:
                for ci in range(0, len(self.pairs), max_batch):
                    chunk = self.pairs[ci:ci + max_batch]
                    energies = _batch_flip_energy_gpu(
                        seq_f, self.lengths,
                        [[(r1, c1), (r2, c2)] for (r1, c1), (r2, c2) in chunk])
                    best_idx = int(xp.argmin(energies))
                    best_pair_e = float(energies[best_idx])
                    if best_pair_e < current_e:
                        (r1, c1), (r2, c2) = chunk[best_idx]
                        self.seq[r1, c1] *= -1
                        self.seq[r2, c2] *= -1
                        seq_f[r1, c1] *= -1
                        seq_f[r2, c2] *= -1
                        current_e = best_pair_e
                        all_flips.extend([(r1, c1), (r2, c2)])
                        improved = True
                        if current_e < 1e-6 or self._exact_energy() == 0:
                            return all_flips, 0.0
                        break

            # Phase 3: Triple evaluation (sampled from gradient top)
            if (step % triple_interval == 0 and current_e > 1e-6
                    and self.n_pos > 10):
                top_pos = self._gradient_top_positions(n_top=50)
                triples = self._sample_triples(top_pos, n_triples=4096)
                if triples:
                    energies = _batch_flip_energy_gpu(
                        seq_f, self.lengths, triples)
                    best_idx = int(xp.argmin(energies))
                    best_trip_e = float(energies[best_idx])
                    if best_trip_e < current_e:
                        for r, c in triples[best_idx]:
                            self.seq[r, c] *= -1
                            seq_f[r, c] *= -1
                            all_flips.append((r, c))
                        current_e = best_trip_e
                        improved = True
                        if current_e < 1e-6 or self._exact_energy() == 0:
                            return all_flips, 0.0

            if not improved:
                break

        return all_flips, current_e


# ── Benchmark ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fixtures import tt_sequences

    rng = np.random.default_rng(2026)

    SOL36 = tt_sequences(36)
    tt8_arr = tt_sequences(8)

    backend = "CUPY" if xp.__name__ == "cupy" else "NUMPY"
    print(f"GPU: {backend}  single+pairs+triples")
    print()

    for name, sol, lengths in [
        ("TT(8)", tt8_arr, np.array((8, 8, 8, 7), dtype=np.int64)),
        ("TT(36)", SOL36, np.array((36, 36, 36, 35), dtype=np.int64)),
    ]:
        n_val = lengths[0]
        total_bits = int(lengths.sum())
        order = 4 * (3 * n_val - 1)
        print(f"{'='*72}")
        print(f"  {name}  n={n_val}  order={order}  bits={total_bits}")
        print(f"{'='*72}")

        for noise_pct in [1, 2, 5, 10, 20]:
            n_flips = max(1, int(total_bits * noise_pct / 100))
            g_rec = gp_rec = 0
            g_time = gp_time = 0.0
            for trial in range(10):
                p = sol.copy()
                for _ in range(n_flips):
                    ri = int(rng.integers(0, 4))
                    ci = int(rng.integers(0, lengths[ri]))
                    p[ri, ci] *= -1

                # Greedy
                g_seq = p.copy()
                t0 = time.perf_counter()
                corr = nonperiodic_autocorrelation_state(
                    g_seq, lengths=lengths, weights=WEIGHTS_I)
                best_e = nonperiodic_correlation_energy(corr)
                imp = True
                while imp and best_e > 0:
                    imp = False
                    for row in range(4):
                        for col in range(int(lengths[row])):
                            ne = apply_nonperiodic_flip(g_seq, corr, row, col,
                                                        lengths=lengths, weight=int(WEIGHTS_I[row]))
                            if ne < best_e:
                                best_e = ne
                                imp = True
                                break
                            apply_nonperiodic_flip(g_seq, corr, row, col,
                                                   lengths=lengths, weight=int(WEIGHTS_I[row]))
                        if imp:
                            break
                if best_e == 0:
                    g_rec += 1
                g_time += time.perf_counter() - t0

                # GPU singles+pairs+triples
                t0 = time.perf_counter()
                gr = GpuRepair(p.copy(), lengths)
                flips, e = gr.repair(pair_interval=3, triple_interval=6)
                if gr._exact_energy() == 0:
                    gp_rec += 1
                gp_time += time.perf_counter() - t0

            print(f"  {noise_pct:>3}% ({n_flips:>2}f) | "
                  f"greedy: {g_rec:>2}/10 ({g_time:.1f}s) | "
                  f"gpu: {gp_rec:>2}/10 ({gp_time:.1f}s)")
