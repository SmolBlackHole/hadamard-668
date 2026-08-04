"""GPU-parallel repair: exact single-flip scan + CPU pair model via Q[i,j]."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from gpu import xp
from correlations import (nonperiodic_autocorrelation_state,
                           nonperiodic_correlation_energy,
                           apply_nonperiodic_flip)

WEIGHTS_I = np.array((1, 1, 2, 2), dtype=np.int64)
WEIGHTS_F = np.array((1, 1, 2, 2), dtype=np.float64)


def _residual_gpu(seq_batch, lengths):
    x_seq = xp.asarray(seq_batch, dtype=xp.float32)
    n = int(lengths[0])
    spectrum = xp.fft.fft(x_seq, n=2 * n - 1, axis=2)
    corr = xp.fft.ifft(xp.abs(spectrum) ** 2, axis=2).real
    w = xp.asarray(WEIGHTS_F, dtype=xp.float32)
    return xp.sum(w[None, :, None] * corr, axis=1)[:, 1:n]


class Repair:
    """GPU singles, CPU pair model via precomputed Q[i,j] interactions."""

    def __init__(self, sequences, lengths):
        self.seq = sequences.copy()
        self.lengths = lengths
        self.n = int(lengths[0])
        self.positions = [(r, c) for r in range(4) for c in range(int(lengths[r]))]
        self.n_pos = len(self.positions)
        self.seq_f = xp.asarray(self.seq.astype(np.float32), dtype=xp.float32)

    def _exact_energy(self):
        return float(nonperiodic_correlation_energy(
            nonperiodic_autocorrelation_state(
                self.seq, lengths=self.lengths, weights=WEIGHTS_I)))

    def _apply(self, idx):
        r, c = self.positions[idx]
        self.seq[r, c] *= -1
        self.seq_f[r, c] *= -1

    def _single_deltas(self):
        B = self.n_pos
        batch = xp.repeat(self.seq_f[None, ...], B, axis=0)
        for i, (r, c) in enumerate(self.positions):
            batch[i, r, c] *= -1
        residuals = _residual_gpu(batch, self.lengths)
        r0 = _residual_gpu(self.seq_f[None, ...], self.lengths)[0]
        return np.array([xp.asnumpy(residuals[i] - r0) for i in range(B)],
                        dtype=np.float64), xp.asnumpy(r0)

    def _pair_interactions(self, top_indices, deltas_top):
        """GPU batch: compute Q[i][j] = residual(i,j) - r0 - delta_i - delta_j."""
        K = len(top_indices)
        if K < 2:
            return np.zeros((K, K, self.n - 1), dtype=np.float64)
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        B = len(pairs)
        batch = xp.repeat(self.seq_f[None, ...], B, axis=0)
        for idx, (pi, pj) in enumerate(pairs):
            r1, c1 = self.positions[top_indices[pi]]
            r2, c2 = self.positions[top_indices[pj]]
            batch[idx, r1, c1] *= -1
            batch[idx, r2, c2] *= -1
        residuals = _residual_gpu(batch, self.lengths)
        r0 = _residual_gpu(self.seq_f[None, ...], self.lengths)[0]
        Q = np.zeros((K, K, self.n - 1), dtype=np.float64)
        for idx, (pi, pj) in enumerate(pairs):
            q = xp.asnumpy(residuals[idx]) - xp.asnumpy(r0) - deltas_top[pi] - deltas_top[pj]
            Q[pi, pj] = q
            Q[pj, pi] = q
        return Q

    def repair(self, max_steps=200, pair_interval=5, pair_top=64):
        all_flips = []
        best_e = self._exact_energy()
        if best_e == 0:
            return [], 0.0

        for step in range(max_steps):
            improved = False

            # Singles: GPU full scan
            deltas, r0 = self._single_deltas()
            energies = np.sum((r0 + deltas) ** 2, axis=1)
            best_idx = int(np.argmin(energies))
            best_single_e = float(energies[best_idx])

            if best_single_e < best_e:
                self._apply(best_idx)
                best_e = self._exact_energy()
                all_flips.append(self.positions[best_idx])
                improved = True
                if best_e == 0:
                    return all_flips, 0.0

            # Pairs: CPU model via Q
            if step % pair_interval == pair_interval - 1 and best_e > 0:
                top_idx = np.argsort(energies)[:pair_top]
                Q = self._pair_interactions(top_idx, deltas[top_idx])
                # Refresh residual for scoring
                r0_now = xp.asnumpy(_residual_gpu(self.seq_f[None, ...], self.lengths)[0])
                best_pair_e = best_e
                best_pair = None
                for i in range(len(top_idx)):
                    for j in range(i + 1, len(top_idx)):
                        e = float(np.sum((r0_now + deltas[top_idx[i]]
                                          + deltas[top_idx[j]] + Q[i, j]) ** 2))
                        if e < best_pair_e:
                            best_pair_e = e
                            best_pair = (top_idx[i], top_idx[j])
                if best_pair is not None and best_pair_e < best_e:
                    pi, pj = best_pair
                    self._apply(pi)
                    self._apply(pj)
                    best_e = self._exact_energy()
                    all_flips.extend([self.positions[pi], self.positions[pj]])
                    improved = True
                    if best_e == 0:
                        return all_flips, 0.0

            if not improved:
                break

        return all_flips, best_e


# ── Benchmark ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fixtures import tt_sequences

    rng = np.random.default_rng(2026)
    backend = "CUPY" if xp.__name__ == "cupy" else "NUMPY"

    for name, sol, lengths in [
        ("TT(8)", tt_sequences(8), np.array((8, 8, 8, 7), dtype=np.int64)),
        ("TT(36)", tt_sequences(36), np.array((36, 36, 36, 35), dtype=np.int64)),
    ]:
        n_val = lengths[0]
        total_bits = int(lengths.sum())
        order = 4 * (3 * n_val - 1)
        print(f"\n{'='*55}")
        print(f"  {name}  n={n_val}  order={order}  bits={total_bits}  ({backend})")
        print(f"{'='*55}")

        for noise_pct in [1, 2, 5, 10, 20]:
            n_flips = max(1, int(total_bits * noise_pct / 100))
            g_rec, gp_rec = 0, 0
            g_t, gp_t = 0.0, 0.0
            for trial in range(10):
                p = sol.copy()
                for _ in range(n_flips):
                    ri = int(rng.integers(0, 4))
                    ci = int(rng.integers(0, lengths[ri]))
                    p[ri, ci] *= -1

                # Greedy
                g_seq = p.copy()
                t0 = time.perf_counter()
                corr = nonperiodic_autocorrelation_state(g_seq, lengths=lengths, weights=WEIGHTS_I)
                best_e = nonperiodic_correlation_energy(corr)
                imp = True
                while imp and best_e > 0:
                    imp = False
                    for row in range(4):
                        for col in range(int(lengths[row])):
                            ne = apply_nonperiodic_flip(
                                g_seq, corr, row, col, lengths=lengths,
                                weight=int(WEIGHTS_I[row]))
                            if ne < best_e: best_e = ne; imp = True; break
                            apply_nonperiodic_flip(
                                g_seq, corr, row, col, lengths=lengths,
                                weight=int(WEIGHTS_I[row]))
                        if imp: break
                if best_e == 0: g_rec += 1
                g_t += time.perf_counter() - t0

                # GPU repair
                t0 = time.perf_counter()
                rp = Repair(p.copy(), lengths)
                flips, e = rp.repair()
                if rp._exact_energy() == 0: gp_rec += 1
                gp_t += time.perf_counter() - t0

            print(f"  {noise_pct:>3}% ({n_flips:>2}f) | "
                  f"greedy: {g_rec:>2}/10 ({g_t:.1f}s) | "
                  f"gpu: {gp_rec:>2}/10 ({gp_t:.1f}s)")
