"""Heatmap visualizations of Hadamard matrices.

Spot patterns: block structure, symmetry, spatial regularity.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fixtures import sylvester, paley
import matplotlib.pyplot as plt

import numpy as np
import matplotlib

matplotlib.use("Agg")


def heatmap(H: np.ndarray, title: str = "", out: str = ""):
    """Standard ±1 heatmap: blue=-1, red=+1."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(H, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{H.shape[1]} columns")
    ax.set_ylabel(f"{H.shape[0]} rows")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig


def gram_heatmap(H: np.ndarray, title: str = "", out: str = ""):
    """H @ H^T — off-diagonal shows where constraints are violated."""
    n = H.shape[0]
    G = H @ H.T.astype(np.float64)
    np.fill_diagonal(G, np.nan)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.abs(G), cmap="hot", interpolation="nearest")
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, label="|dot product|")

    nz = int(np.count_nonzero(G[np.triu_indices(n, k=1)] != 0))
    total = n * (n - 1) // 2
    ax.set_xlabel(f"Non-zero off-diagonal: {nz}/{total}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig


def row_correlation_map(H: np.ndarray, title: str = "", out: str = ""):
    """Which rows correlate with which? Sort by first principal component."""
    n = H.shape[0]
    # PCA for row ordering
    U, S, Vt = np.linalg.svd(H.astype(np.float64), full_matrices=False)
    order = np.argsort(U[:, 0])

    H_sorted = H[order]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Sorted heatmap
    axes[0].imshow(H_sorted, cmap="RdBu_r", vmin=-1,
                   vmax=1, interpolation="nearest")
    axes[0].set_title(f"{title}\nRows sorted by PC1", fontsize=12)

    # Row-row correlation matrix
    G = H @ H.T.astype(np.float64)
    G = G[order][:, order]
    im = axes[1].imshow(np.abs(G), cmap="hot")
    axes[1].set_title("|Row-Row dot product| (sorted)", fontsize=12)
    plt.colorbar(im, ax=axes[1])

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig


def column_structure(H: np.ndarray, title: str = "", out: str = ""):
    """Column-column correlation: find structural symmetries."""
    n = H.shape[0]
    C = H.T @ H.astype(np.float64)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.abs(C), cmap="hot", interpolation="nearest")
    ax.set_title(f"{title}\n|Column-Column dot product|", fontsize=14)
    plt.colorbar(im, ax=ax, label="|dot product|")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig


def block_partition(H: np.ndarray, block_size: int, title: str = "", out: str = ""):
    """Show the matrix as block-wise sums (aggregated)."""
    n = H.shape[0]
    if n % block_size != 0:
        print(
            f"  order {n} not divisible by {block_size}, skipping block view")
        return None
    m = n // block_size
    blocks = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            block = H[i * block_size:(i + 1) * block_size,
                      j * block_size:(j + 1) * block_size]
            blocks[i, j] = int(np.sum(block))

    fig, ax = plt.subplots(figsize=(10, 10))
    vmax = np.abs(blocks).max()
    im = ax.imshow(blocks, cmap="RdBu_r", vmin=-vmax,
                   vmax=vmax, interpolation="nearest")
    ax.set_title(
        f"{title}\n{block_size}x{block_size} block sums ({m}x{m} blocks)", fontsize=14)
    plt.colorbar(im, ax=ax, label="block sum")

    # Annotate blocks with values
    for i in range(m):
        for j in range(m):
            ax.text(j, i, f"{int(blocks[i, j]):+d}", ha="center", va="center",
                    fontsize=6, color="#333333" if abs(blocks[i, j]) < vmax / 2 else "white")

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig


def full_report(name: str, H: np.ndarray, out_dir: str = "plots"):
    """Generate all visualizations for a matrix."""
    import os

    n = H.shape[0]
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{out_dir}/{name}"

    print(f"  {name} ({n}x{n})")
    heatmap(H, f"{name} ({n}x{n})", f"{prefix}_heatmap.png")
    print(f"    heatmap -> {prefix}_heatmap.png")

    gram_heatmap(H, f"{name} Gram", f"{prefix}_gram.png")
    print(f"    gram    -> {prefix}_gram.png")

    col = column_structure(H, f"{name}", f"{prefix}_columns.png")
    if col:
        print(f"    columns -> {prefix}_columns.png")

    row = row_correlation_map(H, f"{name}", f"{prefix}_rows.png")
    if row:
        print(f"    rows    -> {prefix}_rows.png")

    # Block views at interesting sizes
    for bs in [2, 4, 8, n // 4] if n >= 16 else [2]:
        if bs > 0 and n % bs == 0 and bs < n:
            blk = block_partition(
                H, bs, f"{name}", f"{prefix}_blocks_{bs}x{bs}.png")
            if blk:
                print(f"    blocks {bs}x{bs} -> {prefix}_blocks_{bs}x{bs}.png")

    plt.close("all")


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "plots"

    print("=== Sylvester (2^k) ===")
    for n in [4, 8, 16]:
        full_report(f"sylvester_{n:03d}", sylvester(n), out)

    print("\n=== Paley (q+1, q == 3 mod 4) ===")
    for n in [8, 12, 20]:
        full_report(f"paley_{n:03d}", paley(n), out)

    print(f"\nDone. Plots in {out}/")
