# Hadamard Benchmark

Backend: cupy
Seed: 42
Step budgets: n=4: 2000, n=8: 2000, n=12: 2000, n=16: 2000, n=20: 2000, n=668: 5000
Timeout per case: 60 seconds

Cell format: `status; e=energy; rms=root-mean-square correlation; orth=orthogonal pairs; algo=search seconds`.

## Individual strategies

| Strategy | 4 | 8 | 12 | 16 | 20 | 668 |
|---|---|---|---|---|---|---|
| Circulant | YES; algo=0.3s | N/A | YES; algo=0.3s | N/A | YES; algo=0.4s | OK; e=4692032; rms=4.59; orth=175350/222778; algo=0.4s |
| Annealing | YES; algo=0.3s | N/A | YES; algo=0.3s | N/A | YES; algo=0.3s | OK; e=8614528; rms=6.22; orth=174014/222778; algo=0.4s |
| BaumertHall | YES; algo=0.3s | N/A | YES; algo=0.3s | N/A | YES; algo=0.3s | OK; e=11607168; rms=7.22; orth=171342/222778; algo=0.4s |
| Diffset | YES; algo=0.3s | YES; algo=0.3s | YES; algo=0.3s | YES; algo=0.3s | YES; algo=0.3s | OK; e=887104; rms=2.00; orth=167334/222778; algo=0.3s |
| MonteCarlo | YES; algo=0.3s | YES; algo=0.3s | YES; algo=0.3s | YES; algo=0.3s | YES; algo=0.3s | OK; e=855040; rms=1.96; orth=187374/222778; algo=0.9s |
| RepairSearch | YES; algo=0.0s | OK; e=76; rms=1.65; orth=12/28; algo=0.1s | YES; algo=0.0s | OK; e=588; rms=2.21; orth=36/120; algo=0.1s | OK; e=940; rms=2.22; orth=57/190; algo=0.1s | OK; e=132473056; rms=24.39; orth=7393/222778; algo=0.3s |
| Ising | YES; algo=0.4s | YES; algo=0.4s | YES; algo=0.3s | OK; e=128; rms=1.03; orth=112/120; algo=0.4s | OK; e=320; rms=1.30; orth=170/190; algo=0.4s | OK; e=4702720; rms=4.59; orth=172678/222778; algo=0.8s |
| Spectral | YES; algo=0.6s | YES; algo=0.6s | OK; e=192; rms=1.71; orth=54/66; algo=0.7s | OK; e=128; rms=1.03; orth=112/120; algo=0.7s | OK; e=320; rms=1.30; orth=170/190; algo=0.7s | OK; e=2426176; rms=3.30; orth=178690/222778; algo=3.4s |
