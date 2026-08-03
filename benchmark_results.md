# Hadamard Benchmark

Backend: cupy
Seed: 42
Step budgets: n=4: 2000, n=8: 2000, n=12: 2000, n=16: 2000, n=20: 2000, n=668: 5000
Timeout per case: 60 seconds

Cell format: `status; e=energy; rms=root-mean-square correlation; orth=orthogonal pairs; algo=search seconds`.

## Individual strategies

| Strategy | 4 | 8 | 12 | 16 | 20 | 668 |
|---|---|---|---|---|---|---|
| TurynGreedy | N/A | N/A | N/A | N/A | YES; algo=0.2s | OK; e=2479616; rms=3.34; orth=194722/222778; algo=0.4s |
| TurynAnnealing | N/A | N/A | N/A | N/A | YES; algo=0.2s | OK; e=2276544; rms=3.20; orth=192050/222778; algo=0.3s |
| TurynPOCS | N/A | N/A | N/A | N/A | YES; algo=0.2s | OK; e=2158976; rms=3.11; orth=192718/222778; algo=0.7s |
| TurynSteepest | N/A | N/A | N/A | N/A | YES; algo=0.2s | OK; e=3249152; rms=3.82; orth=190046/222778; algo=7.6s |
| MonteCarlo | N/A | N/A | N/A | N/A | YES; algo=2.3s | OK; e=1710080; rms=2.77; orth=194722/222778; algo=5.5s |
| RepairSearch | YES; algo=0.0s | OK; e=76; rms=1.65; orth=12/28; algo=0.0s | YES; algo=0.0s | OK; e=588; rms=2.21; orth=36/120; algo=0.1s | OK; e=940; rms=2.22; orth=57/190; algo=0.0s | OK; e=132473056; rms=24.39; orth=7393/222778; algo=0.3s |
| Spectral | N/A | N/A | N/A | N/A | OK; e=320; rms=1.30; orth=170/190; algo=0.5s | OK; e=1720768; rms=2.78; orth=192050/222778; algo=1.6s |

## Experimental strategies

| Strategy | 4 | 8 | 12 | 16 | 20 | 668 |
|---|---|---|---|---|---|---|
| Genetic | N/A | N/A | N/A | N/A | YES; algo=0.8s | OK; e=2394112; rms=3.28; orth=190046/222778; algo=3.2s |
| Ising | N/A | N/A | N/A | N/A | YES; algo=0.2s | OK; e=4307264; rms=4.40; orth=192718/222778; algo=0.5s |

## Pipelines

| Strategy | 4 | 8 | 12 | 16 | 20 | 668 |
|---|---|---|---|---|---|---|
| TurynGreedy->Repair | N/A | N/A | N/A | N/A | YES; algo=0.2s | OK; e=2479616; rms=3.34; orth=194722/222778; algo=0.4s |
| TurynAnnealing->Repair | N/A | N/A | N/A | N/A | YES; algo=0.2s | OK; e=3046080; rms=3.70; orth=189378/222778; algo=0.4s |
