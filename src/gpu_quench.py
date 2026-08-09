"""GPU volume quench: screen + deep quench + CPU refinement.
RTX 4090 fully utilised — 500+ combos parallel on GPU.
"""

from __future__ import annotations

import time

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def _batch_q_gpu(seqs: np.ndarray) -> np.ndarray:
    """Compute reduced Q for batch via single GPU FFT pipeline."""
    B, n_seqs, n = seqs.shape
    m = (n - 1) // 2
    d_seqs = cp.asarray(seqs, dtype=cp.float32)
    padded = cp.zeros((B, n_seqs, 2 * n), dtype=cp.float32)
    padded[:, :, :n] = d_seqs
    f = cp.fft.rfft(padded, axis=2)
    acf = cp.fft.irfft((cp.abs(f) ** 2).sum(axis=1), n=2 * n, axis=1)
    rev = cp.arange(n - 1, -1, -1, dtype=cp.int32) + 1
    u = cp.round((acf[:, :n] - acf[:, rev])[:, 1:m + 1] / 4.0).astype(cp.int32)
    return cp.asnumpy((u * u).sum(axis=1).astype(cp.int64))


def screen_all_lags_gpu(cur_seq, all_n_arr):
    """Score ALL combos from ALL lags in ONE GPU transfer."""
    n = cur_seq.shape[1]
    combos = []
    for k, n_arr in all_n_arr:
        for i0 in range(len(n_arr[0])):
            for i1 in range(len(n_arr[1])):
                for i2 in range(len(n_arr[2])):
                    for i3 in range(len(n_arr[3])):
                        combos.append((k, int(n_arr[0][i0]) % n, int(n_arr[1][i1]) % n,
                                       int(n_arr[2][i2]) % n, int(n_arr[3][i3]) % n))
    B = len(combos)
    if B == 0:
        return []
    batch = np.tile(cur_seq[None], (B, 1, 1)).astype(np.int8)
    for i, (_, c0, c1, c2, c3) in enumerate(combos):
        batch[i, 0, c0] *= -1; batch[i, 1, c1] *= -1
        batch[i, 2, c2] *= -1; batch[i, 3, c3] *= -1
    t0 = time.perf_counter()
    qs = _batch_q_gpu(batch)
    dt = time.perf_counter() - t0
    order = np.argsort(qs)
    scored = [(int(qs[i]),) + combos[i] for i in order]
    print(f"  [GPU] {B} combos scored in {dt*1000:.0f}ms ({B/dt/1000:.0f}k st/s)  "
          f"Q=[{qs.min()},{qs.max()}]")
    return scored


def gpu_deep_quench(seqs_batch: np.ndarray, max_rounds: int = 30):
    """GPU-batched greedy descent on ALL combos simultaneously."""
    B, n_seqs, n = seqs_batch.shape
    m = (n - 1) // 2
    N = n_seqs * n

    d_seqs = cp.asarray(seqs_batch, dtype=cp.int8)
    d_u = cp.zeros((B, m), dtype=cp.int32)
    d_q = cp.zeros(B, dtype=cp.int64)

    padded = cp.zeros((B, n_seqs, 2 * n), dtype=cp.float32)
    padded[:, :, :n] = d_seqs.astype(cp.float32)
    f = cp.fft.rfft(padded, axis=2)
    acf = cp.fft.irfft((cp.abs(f) ** 2).sum(axis=1), n=2 * n, axis=1)
    rev = cp.arange(n - 1, -1, -1, dtype=cp.int32) + 1
    d_u = cp.round((acf[:, :n] - acf[:, rev])[:, 1:m + 1] / 4.0).astype(cp.int32)
    d_q = (d_u * d_u).sum(axis=1)

    best_q = d_q.copy()
    best_seqs = d_seqs.copy()
    active = cp.ones(B, dtype=cp.bool_)

    for _round in range(max_rounds):
        if not active.any():
            break
        A = int(active.sum())
        act = d_seqs[active]
        au = d_u[active]
        aq = d_q[active]

        # Score all 4n flips via FFT (no delta cache rebuild needed!)
        cands = cp.tile(act[:, None], (1, N, 1, 1)).astype(cp.int8)  # (A, 4n, 4, n)
        cands_flat = cands.reshape(A * N, n_seqs, n)
        flips = cp.arange(N) % n
        seqs_flip = cp.arange(N) // n
        rows = cp.arange(A * N) // N
        cands_flat[cp.arange(A * N), seqs_flip, flips] *= -1

        # FFT batch
        p = cp.zeros((A * N, n_seqs, 2 * n), dtype=cp.float32)
        p[:, :, :n] = cands_flat.astype(cp.float32)
        fi = cp.fft.rfft(p, axis=2)
        tp = (cp.abs(fi) ** 2).sum(axis=1)
        ac = cp.fft.irfft(tp, n=2 * n, axis=1)
        u_new = cp.round((ac[:, :n] - ac[:, rev])[:, 1:m + 1] / 4.0).astype(cp.int32)
        q_new = (u_new * u_new).sum(axis=1).reshape(A, N)

        best_idx = q_new.argmin(axis=1)
        improved = q_new.min(axis=1) < aq

        if not improved.any():
            break

        # Apply flip
        imp_s = best_idx[improved] // n
        imp_c = best_idx[improved] % n
        act[improved, imp_s, imp_c] *= -1

        # Recompute NAF (only for improved, via FFT)
        n_imp = int(improved.sum())
        p2 = cp.zeros((n_imp, n_seqs, 2 * n), dtype=cp.float32)
        p2[:, :, :n] = act[improved].astype(cp.float32)
        fi2 = cp.fft.rfft(p2, axis=2)
        tp2 = (cp.abs(fi2) ** 2).sum(axis=1)
        ac2 = cp.fft.irfft(tp2, n=2 * n, axis=1)
        u2 = cp.round((ac2[:, :n] - ac2[:, rev])[:, 1:m + 1] / 4.0).astype(cp.int32)
        q2 = (u2 * u2).sum(axis=1)

        au[improved] = u2
        aq[improved] = q2

        # Track best
        act_idx = cp.where(active)[0]
        imp_orig = act_idx[improved]
        upd = q2 < best_q[imp_orig]
        best_q[imp_orig[upd]] = q2[upd]
        best_seqs[imp_orig[upd]] = act[improved][upd]

        d_seqs[active] = act
        d_u[active] = au
        d_q[active] = aq
        keep = cp.zeros(B, dtype=cp.bool_)
        keep[act_idx[improved]] = True
        active = keep

    print(f"  [GPU quench] {_round+1}r, best Q={int(best_q.min())}-{int(best_q.max())}")
    return cp.asnumpy(best_q), cp.asnumpy(best_seqs)


def gpu_escape_quench(cur_seq, all_n_arr, rng, budget, best_seq, best_e,
                       search_fn, quench_config, stats):
    """Full GPU: screen ALL combos + GPU deep quench + CPU Tabu on best."""
    scored = screen_all_lags_gpu(cur_seq, all_n_arr)
    if not scored:
        return best_seq, best_e, 0, False

    n_total = len(scored)
    from .tracker import Tracker

    # GPU deep quench on ALL combos (full utilisation, ~1-5s total)
    n_gpu = min(n_total, 1000)
    gpu_batch = np.tile(cur_seq[None], (n_gpu, 1, 1)).astype(np.int8)
    for i in range(n_gpu):
        _, _, c0, c1, c2, c3 = scored[i]
        gpu_batch[i, 0, c0] *= -1; gpu_batch[i, 1, c1] *= -1
        gpu_batch[i, 2, c2] *= -1; gpu_batch[i, 3, c3] *= -1

    gpu_qs, gpu_seqs = gpu_deep_quench(gpu_batch, max_rounds=25)

    # CPU Tabu only on best 10 (Tabu needs Numba kernel)
    total_used = 0
    deep_budget = min(15000, budget // 10)
    gpu_order = np.argsort(gpu_qs)

    for rank in range(min(10, n_gpu)):
        idx = gpu_order[rank]
        cand = gpu_seqs[idx]
        t2 = Tracker(); t2.build(cand)
        sol, be, used, _ = search_fn(cand, t2, rng, steps=deep_budget, config=quench_config)
        total_used += used
        stats.kicks += 1; stats.kick_evals += 1
        if be < best_e:
            best_seq, best_e = sol.copy(), be
        if be == 0:
            return sol, 0, total_used, True

    return best_seq, best_e, total_used, False
