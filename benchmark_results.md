# Hadamard Benchmark

Seed: 42
Timeout per case: 60 seconds

Cell format: `e=energy | orthogonal_pairs/total_pairs | seconds`.

## Individual strategies

| Strategy | 4 | 8 | 12 | 16 | 20 | 668 |
| --- | --- | --- | --- | --- | --- | --- |
| Circulant | YES 1.1s | YES 1.1s | YES 1.1s | YES 1.1s | YES 1.1s | e=5461568 | 174682/222778 | 2.3s |
| Annealing | YES 1.1s | YES 1.1s | YES 1.1s | YES 1.1s | YES 1.1s | e=7289216 | 174014/222778 | 1.9s |
| BaumertHall | YES 0.9s | ERROR | e=384 | 42/66 | 0.9s | ERROR | e=640 | 150/190 | 1.1s | e=48523520 | 73814/222778 | 15.3s |
| Diffset | YES 1.0s | YES 0.9s | e=192 | 54/66 | 0.9s | YES 0.9s | YES 0.9s | e=57448000 | 165330/222778 | 19.1s |
| MonteCarlo | YES 1.1s | YES 1.1s | YES 1.0s | YES 1.0s | e=640 | 150/190 | 1.0s | e=48854848 | 38410/222778 | 6.3s |
| DirectSearch | YES 2.2s | e=80 | 14/28 | 2.4s | e=240 | 27/66 | 2.5s | e=560 | 58/120 | 2.5s | e=1180 | 56/190 | 2.6s | e=141480272 | 7037/222778 | 6.7s |
| RepairSearch | YES 0.9s | e=208 | 10/28 | 2.4s | e=752 | 11/66 | 2.3s | e=1916 | 27/120 | 3.5s | e=3968 | 32/190 | 3.6s | e=148567436 | 7056/222778 | 11.3s |
| Ising | YES 2.7s | YES 2.9s | e=96 | 60/66 | 3.3s | e=192 | 108/120 | 3.1s | e=400 | 138/190 | 3.1s | e=104282332 | 8315/222778 | 13.6s |
| Spectral | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT |
