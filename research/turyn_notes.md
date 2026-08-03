# Turyn-Type Construction

## What Turyn gives us

Turyn-type sequences are a DIFFERENT mathematical construction than Williamson/GS.
They produce Hadamard matrices of order 4m where m is odd.

For 668 = 4 x 167:

- Build 4 sequences X,Y,Z,W with lengths (56,56,56,55)
- Non-periodic autocorrelation: N_X(j) + N_Y(j) + 2*N_Z(j) + 2*N_W(j) = 0 for all j>=1
- These fold into "T-sequences" of length 2*(56+56)+55-1 = 222 (approx)
- T-sequences -> base sequences (167) -> Goethals-Seidel -> H(668)

Why this matters:

- 223 free bits (vs 336 in symmetric GS)
- SMALLER search space, DIFFERENT constraints
- This is the route that solved order 428 (4x107)
- jdbrande's framework runs this as PRIMARY route with 2:1 priority over GS

## What we need to implement

```python
# construtions.py
def turyn_npac(x, y, z, w) -> int:
    """Non-periodic autocorrelation energy for Turyn-type sequences.
    N_X(j) + N_Y(j) + 2*N_Z(j) + 2*N_W(j) = 0 for all j>=1
    """
def build_turyn(x, y, z, w, lengths=(56,56,56,55)):
    """Turyn-type -> T-sequences -> base sequences -> GS -> H(668)"""
```

## Why POCS/Ising could work here

- Fewer variables (223 vs 336) = easier search
- Non-periodic autocorrelation = DIFFERENT energy landscape
- FFT can still compute non-periodic autocorrelation via zero-padding
- Ising/Mean-Field works on any Gram-like energy function

## What we DON'T have yet

The exact folding from TT(56) -> T-sequences -> base sequences -> GS.
This is in jdbrande's `core.py`. We'd need to either:
a) Reimplement from mathematical literature
b) Extract from their open-source code
c) Build our own simplified version
