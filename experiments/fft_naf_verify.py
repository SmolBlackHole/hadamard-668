"""Verify: NAF via FFT matches Tracker._compute_residual exactly.

Key identity: NAF_a[t] = ACF_a[t] - ACF_a[n-t]
where ACF is the standard aperiodic autocorrelation, computable via
zero-padding to 2n + FFT.
"""

import numpy as np
import sys
sys.path.insert(0, ".")


def naf_direct(seqs: np.ndarray) -> np.ndarray:
    """Negacyclic ACF sum (same as Tracker._compute_residual)."""
    n = seqs.shape[1]
    j = np.arange(n, dtype=np.intp)[:, None]
    t = np.arange(n, dtype=np.intp)[None, :]
    shifted = (j + t) % n
    sign = np.where(j + t < n, 1, -1).astype(np.int32)
    values = seqs.astype(np.int32)
    return np.sum(values[:, :, None] * values[:, shifted] * sign, axis=(0, 1))


def naf_fft(seqs: np.ndarray) -> np.ndarray:
    """Negacyclic ACF sum via FFT (zero-pad + IFFT)."""
    n = seqs.shape[1]
    pad_len = 2 * n
    total_power = np.zeros(pad_len, dtype=np.float64)

    for s in range(seqs.shape[0]):
        a = seqs[s].astype(np.float64)
        padded = np.zeros(pad_len, dtype=np.float64)
        padded[:n] = a
        f = np.fft.rfft(padded)
        power = np.real(np.fft.irfft(np.abs(f) ** 2, n=pad_len))
        total_power += power

    # total_power[t] = Σ_s ACF_s[t]  (standard aperiodic ACF)
    # Key identity: NAF[t] = ACF[t] - ACF[n-t] = c[t] - c[n-t]
    acf_sum = total_power
    naf = np.zeros(n, dtype=np.float64)
    for t in range(n):
        naf[t] = acf_sum[t] - acf_sum[n - t]  # n-t maps ACF[n-t], anti-symmetry
    return naf


def naf_fft_pytorch(seqs: np.ndarray) -> np.ndarray:
    """Same as naf_fft but with PyTorch (GPU-ready)."""
    import torch
    n = seqs.shape[1]
    pad_len = 2 * n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    a = torch.from_numpy(seqs).to(dtype=torch.float32, device=device)
    total_power = torch.zeros(pad_len, dtype=torch.float32, device=device)

    for s in range(seqs.shape[0]):
        padded = torch.zeros(pad_len, dtype=torch.float32, device=device)
        padded[:n] = a[s]
        f = torch.fft.rfft(padded)
        power = torch.fft.irfft(torch.abs(f) ** 2, n=pad_len)
        total_power += power

    acf_sum = total_power
    naf = torch.zeros(n, dtype=torch.float32, device=device)
    for t in range(n):
        naf[t] = acf_sum[t] - acf_sum[(2 * n - t) % (2 * n)]
    return naf.cpu().numpy()


def main():
    rng = np.random.default_rng(42)
    ns = [11, 17, 23, 31, 41, 47, 53, 67, 83]

    print("Verifying NAF_FFT = NAF_direct (Tracker equivalent)")
    print("=" * 70)

    for n in ns:
        seqs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, n))

        direct = naf_direct(seqs)
        fft_result = naf_fft(seqs)

        diff = np.max(np.abs(direct.astype(np.float64) - fft_result))
        m = (n - 1) // 2
        u_direct = direct[1:m + 1] // 4
        u_fft = (fft_result[1:m + 1] / 4.0).round().astype(np.int32)

        u_match = np.allclose(u_direct, u_fft)
        status = "OK" if (diff < 1.0 and u_match) else "FAIL"
        print(f"n={n:>3}: max|delta|={diff:6.1f}  u_match={u_match}  {status}")

    # Also verify via Tracker
    print("\nVerifying against Tracker.build()")
    print("-" * 40)

    from src.tracker import Tracker

    for n in [23, 41, 67]:
        seqs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, n))
        t = Tracker()
        t.build(seqs)
        m = (n - 1) // 2

        fft_result = naf_fft(seqs)
        u_from_fft = (fft_result[1:m + 1] / 4.0).round().astype(np.int32)

        assert t._u is not None
        match = np.array_equal(t._u, u_from_fft)
        diff = np.max(np.abs(t._u.astype(np.float64) - u_from_fft.astype(np.float64)))
        print(f"n={n:>3}: Tracker.u vs FFT.u  max|delta|={diff}  match={match}")

    # Test PyTorch version
    print("\nVerifying PyTorch FFT")
    print("-" * 20)
    try:
        for n in [17, 31, 53]:
            seqs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, n))
            np_result = naf_fft(seqs)
            pt_result = naf_fft_pytorch(seqs)
            diff = np.max(np.abs(np_result.astype(np.float64) - pt_result.astype(np.float64)))
            print(f"n={n:>3}: NumPy vs PyTorch max|delta|={diff:.4f}  {'OK' if diff < 0.1 else 'FAIL'}")
    except ImportError:
        print("PyTorch not available, skipping")

    print("\nDone.")


if __name__ == "__main__":
    main()
