# NGP dataset from Balonin & Djokovic (2015)

This folder contains the NG-pairs printed in:

- Appendix C: second Paley series
- Appendix D: Ito series

of *Negaperiodic Golay pairs and Hadamard matrices* (arXiv:1508.00640v1).

The paper prints truncated sequences for length `v = 2t`:

- `a[0..t]`, with `a_i = a_{v-i}` for `1 <= i < v`
- `b[0..t-1]`, with `b_i = -b_{v-1-i}`

`ngp_pairs.json` contains both the printed truncated values and the fully reconstructed
`a` and `b` arrays.

## Files

- `ngp_pairs.json`: 41 source records with complete sequences and metadata
- `ngp_pairs.csv`: flattened version for quick inspection
- `ngp_dataset.py`: loader, NAF calculation, 2N builder and validators

There are 17 second-Paley records and 24 Ito records. Lengths occurring in both
families are intentionally retained as separate records.

## Usage

```python
from ngp_dataset import get_pair, get_pairs, build_2n_negaperiodic, validate_pair

a, b = get_pair(30, family="second_paley")
validate_pair(a, b)

H = build_2n_negaperiodic(a, b)
print(H.shape)  # (60, 60)

# Both source families where available:
records = get_pairs(34)
for record in records:
    print(record["family"], record["id"])
```

Validate the complete dataset:

```bash
python ngp_dataset.py
```

Expected result:

```text
Validated 41/41 NGP fixtures.
```

Every included record was checked for:

1. `NAF_a(k) + NAF_b(k) = 0` for all nonzero lags
2. `H H^T = 2v I` for `H = [[A, B], [-B.T, A.T]]`
