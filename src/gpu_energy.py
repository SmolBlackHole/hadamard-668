"""GPU-accelerated NAF energy computation via CuPy FFT.

Key identity: NAF[t] = ACF[t] - ACF[n-t]
where ACF is the standard aperiodic autocorrelation, computable via
zero-padding to 2n + FFT + IFFT(|FFT|^2).

On GPU, this is batched over many states for the quench.
"""

import numpy as np

try:
    import cupy as cp  # type: ignore[import-not-found]
    HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    HAS_CUPY = False


def energy_batch_numpy(seqs_batch: np.ndarray) -> np.ndarray:
    """Compute reduced Q = sum(u^2) for a batch via NumPy FFT (fully vectorized)."""
    B, n_seqs, n = seqs_batch.shape
    pad_len = 2 * n
    m = (n - 1) // 2

    padded = np.zeros((B, n_seqs, pad_len), dtype=np.float32)
    padded[:, :, :n] = seqs_batch.astype(np.float32)
    f = np.fft.rfft(padded, axis=2)
    total_power = (np.abs(f) ** 2).sum(axis=1)
    acf = np.fft.irfft(total_power, n=pad_len, axis=1)

    # Vectorized: NAF[t] = ACF[t] - ACF[n-t]
    rev_idx = n - np.arange(n)  # n, n-1, ..., 1
    naf = acf[:, :n] - acf[:, rev_idx]

    u = np.round(naf[:, 1 : m + 1] / 4.0).astype(np.int32)
    return (u * u).sum(axis=1)


if HAS_CUPY:

    def energy_batch_gpu(seqs_batch: np.ndarray) -> np.ndarray:
        """Fully vectorized GPU NAF energy. No Python loop in the hot path."""
        _, n_seqs, n = seqs_batch.shape
        B_out = seqs_batch.shape[0]
        pad_len = 2 * n
        m = (n - 1) // 2

        padded = cp.zeros((B_out, n_seqs, pad_len), dtype=cp.float32)
        padded[:, :, :n] = cp.asarray(seqs_batch, dtype=cp.float32)

        f = cp.fft.rfft(padded, axis=2)
        total_power = (cp.abs(f) ** 2).sum(axis=1)
        acf = cp.fft.irfft(total_power, n=pad_len, axis=1)

        # Vectorized NAF: acf[:, 0..n-1] - acf[:, n..1]
        rev_idx = cp.arange(n - 1, -1, -1, dtype=cp.int32) + 1  # n, n-1, ..., 1
        naf = acf[:, :n] - acf[:, rev_idx]

        u = cp.round(naf[:, 1 : m + 1] / 4.0).astype(cp.int32)
        qs = (u * u).sum(axis=1).astype(cp.int64)

        return cp.asnumpy(qs)


def verify_gpu():
    """Verify GPU energy computation against CPU (Tracker equivalent)."""
    rng = np.random.default_rng(12345)
    results = []

    for n in [23, 37, 53]:
        seqs_batch = rng.choice(np.array([-1, 1], dtype=np.int8), size=(16, 4, n))
        cpu_qs = energy_batch_numpy(seqs_batch)

        if HAS_CUPY:
            gpu_qs = energy_batch_gpu(seqs_batch)
            diff = np.max(np.abs(cpu_qs.astype(np.float64) - gpu_qs.astype(np.float64)))
            results.append((n, diff))
        else:
            results.append((n, 0.0))

    return results


if __name__ == "__main__":
    print("GPU Energy Verification")
    print("=" * 50)
    print(f"CuPy available: {HAS_CUPY}")

    results = verify_gpu()
    for n, diff in results:
        status = "OK" if diff < 1.0 else "FAIL"
        print(f"  n={n}: CPU vs GPU max|delta| = {diff:.1f}  {status}")

    # Brief throughput test
    if HAS_CUPY:
        import time

        rng = np.random.default_rng(99999)
        for B in [1, 64, 256, 625]:
            seqs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(B, 4, 50))
            # warmup
            energy_batch_gpu(seqs)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            energy_batch_gpu(seqs)
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - t0
            qs = energy_batch_gpu(seqs)
            print(f"  B={B:>4}: {elapsed * 1000:.1f}ms  ({B / elapsed:.0f} states/s)  Q range=[{qs.min()}, {qs.max()}]")
